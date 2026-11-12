"""
Evaluate the K-fold ensemble on the test set (Tables 1 and 3, Fig. 6).

    python evaluate.py --config configs/layer_1000m.yaml --checkpoints checkpoints/layer_1000m
    python evaluate.py --config configs/envelope.yaml   --checkpoints checkpoints/envelope --scatter

The final prediction is the mean of the K fold models' predictions (each
de-normalised with its own training-set scale). Statistics are computed over
    envelope configs : cloudy pixels only (background excluded)
    layer configs    : "unmasked" = all pixels, "masked" = cloudy pixels of the
                       middle frame, dilated by 5x5 (Table 1 reports both)
and written to <output-dir>/eval_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vertical_airflow.dataset import NPZSequenceDataset
from vertical_airflow.utils import find_fold_checkpoint, fold_dirs, load_config, load_model_from_checkpoint

MIDDLE_FRAME, DILATION = 5, 5


def apply_gain_mismatch(x, rel_gain, camera_index=1):
    """Relative radiometric mismatch between the stereo cameras: x_cam *= (1 + rel_gain)."""
    if rel_gain and x.shape[2] > camera_index:
        x = x.clone()
        x[:, :, camera_index] = x[:, :, camera_index] * (1.0 + rel_gain)
    return x


def eval_masks(mask, mode):
    """Boolean pixel selections [B, T, 1, H, W] for each requested statistic."""
    out = {}
    if mode == "envelope":
        out["masked"] = mask > 0.1
    else:
        out["unmasked"] = torch.ones_like(mask, dtype=torch.bool)
        m = F.max_pool2d(mask[:, MIDDLE_FRAME], kernel_size=DILATION, stride=1, padding=DILATION // 2)
        out["masked"] = m.unsqueeze(1).expand_as(mask) > 0.1
    return out


def summarize(diff):
    diff = diff.astype(np.float64)
    return {"MAE": float(np.mean(np.abs(diff))), "ME": float(np.mean(diff)), "STD": float(np.std(diff)),
            "RMSE": float(np.sqrt(np.mean(diff ** 2))), "N": int(diff.size)}


def balanced_scatter(gt, pred, bin_width=0.02, per_bin=5, lim=(-8.5, 8.5), seed=0):
    """Sample up to ``per_bin`` points per GT-velocity bin (Fig. 6)."""
    rng = np.random.default_rng(seed)
    idx = np.digitize(gt, np.arange(lim[0], lim[1] + bin_width, bin_width))
    pick = [rng.choice(np.where(idx == b)[0], size=min(per_bin, (idx == b).sum()), replace=False) for b in np.unique(idx)]
    pick = np.concatenate(pick) if pick else np.arange(gt.size)
    return gt[pick], pred[pick]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", required=True, help="directory with one checkpoint per fold")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--test-npz", default=None, help="evaluate on another npz (e.g. a cross-domain sequence)")
    ap.add_argument("--gain-mismatch", type=float, default=0.0,
                    help="relative gain applied to camera B at test time (e.g. 0.03)")
    ap.add_argument("--scatter", action="store_true", help="save a GT-vs-inferred scatter PDF and the sampled points")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output-dir", default=None, help="where to write results (default: --checkpoints)")
    args = ap.parse_args()
    out_dir = args.output_dir or args.checkpoints
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(args.data_root, cfg["data"]["kfold_dir"])
    one_sat = bool(cfg["data"].get("use_one_satellite", False))
    mode = cfg["data"].get("target", "layer")

    models, norms = [], []
    for fold in fold_dirs(root):
        ck = find_fold_checkpoint(args.checkpoints, fold)
        if ck is None:
            print(f"[eval] no checkpoint for {fold}, skipped")
            continue
        model, _ = load_model_from_checkpoint(ck, device)
        models.append(model)
        norms.append(NPZSequenceDataset(os.path.join(root, fold, "train_w.npz"), one_sat, verbose=False))
        print(f"[eval] loaded {ck}")
    if not models:
        raise SystemExit("no fold checkpoints found")

    test_ds = NPZSequenceDataset(args.test_npz or os.path.join(root, "test_w.npz"), one_sat).use_normalization_of(norms[0])
    diffs = {}
    gts, preds = [], []
    with torch.no_grad():
        for x, y, mask in DataLoader(test_ds, batch_size=args.batch_size, shuffle=False):
            x = apply_gain_mismatch(x, args.gain_mismatch).to(device)
            fold_preds = [ds.denormalize(m(x)[0].cpu().numpy()) for m, ds in zip(models, norms)]
            pred = np.mean(np.stack(fold_preds), axis=0)
            gt = test_ds.denormalize(y.numpy())
            for key, sel in eval_masks(mask, mode).items():
                s = sel.numpy()
                diffs.setdefault(key, []).append((pred - gt)[s])
                if key == "masked":
                    gts.append(gt[s]); preds.append(pred[s])

    results = {k: summarize(np.concatenate(v)) for k, v in diffs.items()}
    print(f"\n[eval] {cfg['name']} - {len(models)}-model ensemble on {len(test_ds)} sequences")
    for k, r in results.items():
        print(f"  {k:9s} MAE={r['MAE']:.2f}  ME={r['ME']:+.2f}  STD={r['STD']:.2f}  RMSE={r['RMSE']:.2f}  (N={r['N']:,})")
    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    if args.scatter:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g, p = balanced_scatter(np.concatenate(gts), np.concatenate(preds))
        np.savez(os.path.join(out_dir, "scatter_points.npz"), gt=g, pred=p)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(g, p, s=2, alpha=0.5)
        lim = max(np.abs(g).max(), np.abs(p).max())
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlabel("Ground Truth [m/s]"); ax.set_ylabel("Inferred [m/s]"); ax.set_aspect("equal")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "scatter.pdf"))
        print(f"[eval] scatter saved to {out_dir}/scatter.pdf")


if __name__ == "__main__":
    main()
