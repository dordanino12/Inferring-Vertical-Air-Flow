"""
Train the spatiotemporal model with K-fold cross-validation (Sec. 3.5-4.1).

    python train.py --config configs/envelope.yaml
    python train.py --config configs/layer_1000m.yaml --folds fold_01_val_r0-1_c0-2

For every fold_* directory under the dataset root one model is trained on
train_w.npz, selected on val_w.npz and evaluated on the shared test_w.npz.
Checkpoints are written to <output_dir>/<name>/<fold>.pt. Ensemble evaluation
of the K models is done by evaluate.py.

Training procedure (as in the paper's released models):
  Stage 1  encoder frozen,    batch 32, AdamW lr 1e-3, ReduceLROnPlateau(patience 3)
  Stage 2  encoder trainable, batch 16, AdamW lr 3e-4, ReduceLROnPlateau(patience 5)
  Loss     power loss (epochs 1-10) -> linear blend (11-15) -> bin loss (16+)
  The checkpoint with the lowest validation bin loss within Stage 2 is kept
  (the best Stage-1 bin-loss weights, if any, initialise Stage 2).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from vertical_airflow.dataset import NPZSequenceDataset
from vertical_airflow.losses import LossSchedule
from vertical_airflow.utils import build_model, fold_dirs, load_config, set_seed


def run_epoch(model, loader, device, loss_fn, epoch, use_mask, dataset, optimizer=None, scaler=None, grad_clip=1.0):
    """One pass over ``loader``; trains if an optimizer is given. Returns loss, MAE, RMSE, ME [m/s]."""
    train = optimizer is not None
    model.train(train)
    tot_loss, n = 0.0, 0
    s_abs = s_sq = s_err = 0.0
    cnt = 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                y_pred, _ = model(x)
            loss = loss_fn(y_pred, y, mask, epoch)
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        tot_loss += loss.detach().item() * x.size(0)
        n += x.size(0)

        with torch.no_grad():  # metrics in m/s, on cloudy pixels if a mask is used
            diff = (dataset.denormalize(y_pred).cpu().numpy() - dataset.denormalize(y).cpu().numpy())
            if use_mask:
                diff = diff[mask.cpu().numpy().astype(bool)]
            diff = diff.astype(np.float64).ravel()
            s_abs += np.abs(diff).sum(); s_sq += (diff ** 2).sum(); s_err += diff.sum(); cnt += diff.size
    if cnt == 0:
        return tot_loss / n, 0.0, 0.0, 0.0
    return tot_loss / n, s_abs / cnt, float(np.sqrt(s_sq / cnt)), s_err / cnt


def train_fold(cfg, train_path, val_path, test_path, out_path, device, status=None):
    """``status(epoch, total, stage, val_loss, best_val, sec_per_epoch)`` is called after every epoch, if given."""
    tcfg, mcfg, lcfg = cfg["train"], cfg["model"], cfg["loss"]
    one_sat = bool(cfg["data"].get("use_one_satellite", False))
    use_mask = lcfg["use_mask"]

    train_ds = NPZSequenceDataset(train_path, one_sat)
    val_ds = NPZSequenceDataset(val_path, one_sat).use_normalization_of(train_ds)
    test_ds = NPZSequenceDataset(test_path, one_sat).use_normalization_of(train_ds)
    bin_min, bin_max = float(np.min(train_ds.Y)), float(np.max(train_ds.Y))
    print(f"[train] bins over [{bin_min:.4f}, {bin_max:.4f}] m/s | train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")

    def loaders(bs):
        return (DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True),
                DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True),
                DataLoader(test_ds, batch_size=bs, shuffle=False, pin_memory=True))

    model = build_model(mcfg["backbone"], train_ds.C, mcfg.get("use_conv_lstm", True), freeze_encoder=True).to(device)
    loss_fn = LossSchedule(lcfg, train_ds, bin_min, bin_max)

    def set_encoder_trainable(flag):
        for p in model.encoder.parameters():
            p.requires_grad = flag

    def new_optim(lr, patience):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=tcfg["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=patience)
        return opt, sched, GradScaler(enabled=(device.type == "cuda"))

    e1, e2 = int(tcfg["epochs_stage1"]), int(tcfg["epochs_stage2"])
    set_encoder_trainable(False)
    train_loader, val_loader, test_loader = loaders(tcfg["batch_size_stage1"])
    optimizer, scheduler, scaler = new_optim(tcfg["lr_stage1"], tcfg["plateau_patience_stage1"])
    best_val, best_state, stage = float("inf"), None, 1
    t_start = time.time()

    for epoch in range(1, e1 + e2 + 1):
        if epoch == e1 + 1:  # ---- Stage 1 -> Stage 2 ----
            if best_state is not None:
                model.load_state_dict(best_state)
            stage = 2
            set_encoder_trainable(True)
            train_loader, val_loader, test_loader = loaders(tcfg["batch_size_stage2"])
            optimizer, scheduler, scaler = new_optim(tcfg["lr_stage2"], tcfg["plateau_patience_stage2"])
            best_val, best_state = float("inf"), None
            print(f"[train] Stage 2: encoder trainable, "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

        tr = run_epoch(model, train_loader, device, loss_fn, epoch, use_mask, train_ds, optimizer, scaler, tcfg["grad_clip"])
        va = run_epoch(model, val_loader, device, loss_fn, epoch, use_mask, train_ds)
        scheduler.step(va[0])
        mode, _ = loss_fn.mode(epoch)
        print(f"Epoch {epoch}/{e1 + e2} (Stage {stage}, loss={mode})\n"
              f"  Train: Loss={tr[0]:.4f} | MAE={tr[1]:.4f} | RMSE={tr[2]:.4f} | ME={tr[3]:.4f}\n"
              f"  Val:   Loss={va[0]:.4f} | MAE={va[1]:.4f} | RMSE={va[2]:.4f} | ME={va[3]:.4f}", flush=True)

        if loss_fn.is_final_loss(epoch) and va[0] < best_val:
            best_val, best_state = va[0], copy.deepcopy(model.state_dict())
            torch.save({
                "model_state": model.state_dict(),
                "config": {"type": mcfg["backbone"], "in_channels": train_ds.C, "use_one_satellite": one_sat,
                           "use_conv_lstm": bool(mcfg.get("use_conv_lstm", True)), "stage": stage},
                "train_config": cfg,
                "val_loss": best_val, "epoch": epoch, "stage": stage,
            }, out_path)
            print(f"   -> new best (val loss {best_val:.4f}), saved {out_path}")
        if status:
            status(epoch, e1 + e2, stage, va[0], best_val, (time.time() - t_start) / epoch)

    if best_state is not None:
        model.load_state_dict(best_state)
    te = run_epoch(model, test_loader, device, loss_fn, e1 + e2, use_mask, train_ds)
    print(f"Test:  Loss={te[0]:.4f} | MAE={te[1]:.4f} | RMSE={te[2]:.4f} | ME={te[3]:.4f}")
    return {"best_val_loss": best_val, "test_loss": te[0], "test_mae": te[1], "test_rmse": te[2], "test_me": te[3]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="data", help="directory that contains the k-fold dataset folders")
    ap.add_argument("--output-dir", default="checkpoints")
    ap.add_argument("--folds", nargs="*", help="train only these fold directories (default: all)")
    ap.add_argument("--seed", type=int, default=None, help="overrides train.seed in the config")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip folds already finished (listed in train_metrics.json); use to resume an interrupted run")
    ap.add_argument("--epochs", type=int, nargs=2, metavar=("STAGE1", "STAGE2"), default=None,
                    help="override the number of epochs of each stage (e.g. 1 1 for a quick smoke test)")
    ap.add_argument("--status-file", default=None, help="one-line progress file rewritten after every epoch")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs_stage1"], cfg["train"]["epochs_stage2"] = args.epochs
    set_seed(args.seed if args.seed is not None else cfg["train"].get("seed"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(args.data_root, cfg["data"]["kfold_dir"])
    test_path = os.path.join(root, "test_w.npz")
    out_dir = os.path.join(args.output_dir, cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] {cfg['name']} on {device} | data {root} | output {out_dir}")

    metrics_path = os.path.join(out_dir, "train_metrics.json")
    results = {}
    if args.skip_existing and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            results = json.load(f)
    folds = args.folds or fold_dirs(root)
    for k, fold in enumerate(folds, 1):
        if fold in results and os.path.exists(os.path.join(out_dir, f"{fold}.pt")):
            print(f"[train] {fold} already finished, skipping")
            continue
        print(f"\n{'=' * 70}\n[train] {fold}\n{'=' * 70}")

        def status(epoch, total, stage, val_loss, best_val, sec_per_epoch, fold=fold, k=k):
            if not args.status_file:
                return
            eta = time.strftime("%a %H:%M", time.localtime(time.time() + (total - epoch) * sec_per_epoch))
            line = (f"{time.strftime('%Y-%m-%d %H:%M')} | {cfg['name']} | fold {k}/{len(folds)} ({fold}) | "
                    f"epoch {epoch}/{total} (stage {stage}) | val {val_loss:.4f} | best {best_val:.4f} | "
                    f"{sec_per_epoch / 60:.1f} min/epoch | fold ends ~{eta}\n")
            with open(args.status_file, "w") as f:
                f.write(line)

        results[fold] = train_fold(cfg, os.path.join(root, fold, "train_w.npz"), os.path.join(root, fold, "val_w.npz"),
                                   test_path, os.path.join(out_dir, f"{fold}.pt"), device, status)
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n[train] per-fold test metrics (single models; use evaluate.py for the ensemble):")
    for fold, m in results.items():
        print(f"  {fold}: MAE={m['test_mae']:.4f} RMSE={m['test_rmse']:.4f} ME={m['test_me']:.4f}")


if __name__ == "__main__":
    main()
