"""
Fig. 6 - inferred vs. ground-truth V_z on random test pixels, several experiments overlaid.

Each run is  LABEL=CONFIG:CHECKPOINT_DIR[:COLOR]. Points are sampled uniformly over the
ground-truth velocity (up to --points-per-bin per 0.02 m/s bin) so that rare, strong
velocities are visible. Envelope runs use cloudy pixels only, layer runs all pixels.

    # Fig. 6 left
    python figures/fig6_scatter.py --output results/fig6_left.pdf --runs \
        "Z = 1000m=configs/layer_1000m.yaml:checkpoints/layer_1000m:tab:red" \
        "Z = 1500m=configs/layer_1500m.yaml:checkpoints/layer_1500m:tab:green" \
        "Envelope=configs/envelope.yaml:checkpoints/envelope:tab:blue"
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vertical_airflow.ensemble import Ensemble  # noqa: E402

BIN_WIDTH = 0.02
VEL_RANGE = (-15.0, 15.0)


def parse_run(spec):
    label, rest = spec.rsplit("=", 1)
    parts = rest.split(":")
    color = ":".join(parts[2:]) if len(parts) > 2 else None
    return label, parts[0], parts[1], color


def collect(ens, batch_size):
    """All evaluated (gt, pred) pixel pairs of the test set."""
    ds = ens.test_set()
    gts, preds = [], []
    for x, y, mask in tqdm(DataLoader(ds, batch_size=batch_size), desc=ens.cfg["name"]):
        pred, _ = ens.predict(x)
        gt = ds.denormalize(y.numpy())
        sel = (mask.numpy() > 0.1) if ens.target == "envelope" else np.ones(gt.shape, bool)
        gts.append(gt[sel]); preds.append(pred[sel])
    return np.concatenate(gts), np.concatenate(preds)


def balanced_sample(gt, pred, per_bin, rng):
    idx = np.digitize(gt, np.arange(VEL_RANGE[0], VEL_RANGE[1] + BIN_WIDTH, BIN_WIDTH))
    pick = [rng.choice(np.where(idx == b)[0], size=min(per_bin, int((idx == b).sum())), replace=False)
            for b in np.unique(idx)]
    pick = np.concatenate(pick)
    rng.shuffle(pick)
    return gt[pick], pred[pick]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output", default="results/fig6.pdf")
    ap.add_argument("--title", default=None)
    ap.add_argument("--points-per-bin", type=int, default=5)
    ap.add_argument("--lim", type=float, default=8.5, help="axis limit [m/s]")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    saved = {}
    for spec in args.runs:
        label, cfg, ckpt, color = parse_run(spec)
        gt, pred = collect(Ensemble(cfg, ckpt, args.data_root), args.batch_size)
        g, p = balanced_sample(gt, pred, args.points_per_bin, rng)
        saved[label] = np.stack([g, p])
        ax.scatter(g, p, s=4, alpha=0.4, color=color, label=label, rasterized=True)
    lim = args.lim
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Ground Truth [m/s]", fontweight="bold"); ax.set_ylabel("Inferred [m/s]", fontweight="bold")
    if args.title:
        ax.text(0.04, 0.96, args.title, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    ax.legend(loc="lower right", markerscale=4)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.output, bbox_inches="tight")
    np.savez(os.path.splitext(args.output)[0] + "_points.npz", **saved)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
