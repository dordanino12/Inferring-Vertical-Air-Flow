"""
Fig. 7 - qualitative results: input image, ground-truth V_z and inferred V_z.

One row per run (LABEL=CONFIG:CHECKPOINT_DIR). The K-fold ensemble sees the whole
12-frame sequence; --frame selects the frame shown. Each panel is also saved as a
separate PDF (as used to compose the paper figure).

    python figures/fig7_qualitative.py --sequence 100 --frame 5 --output-dir results/fig7 --runs \
        "Envelope=configs/envelope.yaml:checkpoints/envelope" \
        "Z = 1000m=configs/layer_1000m.yaml:checkpoints/layer_1000m" \
        "Z = 1500m=configs/layer_1500m.yaml:checkpoints/layer_1500m"
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vertical_airflow.ensemble import Ensemble  # noqa: E402

LINTHRESH = 2.0   # m/s, linear part of the symmetric-log colour scale
TICK_STEP = 2.0   # m/s, colour-bar tick spacing
GAMMA = 0.5


def gamma(img):
    lo, hi = img.min(), img.max()
    return np.zeros_like(img) if hi - lo < 1e-6 else ((img - lo) / (hi - lo)) ** GAMMA


def velocity_cmap():
    cmap = mcolors.ListedColormap(plt.get_cmap("jet", 256)(np.linspace(0, 1, 256)), name="jet_custom")
    cmap.set_bad("black")
    return cmap


def velocity_ticks(norm):
    ticks = np.arange(norm.vmin, norm.vmax + 1e-6, TICK_STEP)
    if norm.vmin < 0 < norm.vmax:          # add 0 and drop ticks crowding it
        ticks = np.sort(np.append(ticks[np.abs(ticks) >= 0.5 * TICK_STEP], 0.0))
    return ticks


def add_colorbar(fig, im, ax, norm):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_ticks(velocity_ticks(norm))
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    cb.ax.tick_params(labelsize=10)
    return cb


def save_panel(img, path, cmap="gray", norm=None, colorbar=False):
    fig, ax = plt.subplots(figsize=(6, 5.2) if colorbar else (5, 5), dpi=200)
    im = ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
    ax.axis("off")
    if colorbar:
        add_colorbar(fig, im, ax, norm)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sequence", type=int, default=100, help="index in test_w.npz")
    ap.add_argument("--frame", type=int, default=5, help="frame 0..11")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output-dir", default="results/fig7")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cmap = velocity_cmap()

    rows = []
    for spec in args.runs:
        label, rest = spec.rsplit("=", 1)
        cfg, ckpt = rest.split(":")[:2]
        ens = Ensemble(cfg, ckpt, args.data_root)
        ds = ens.test_set()
        x, y, mask = ds[args.sequence]
        pred, _ = ens.predict(x[None])
        t = args.frame
        img = gamma(x[t, 0].numpy())
        gt = ds.denormalize(y.numpy())[t, 0]
        pr = pred[0, t, 0]
        norm = mcolors.SymLogNorm(linthresh=LINTHRESH, linscale=1.0, vmin=float(ds.Y.min()), vmax=float(ds.Y.max()))
        if ens.target == "envelope":                     # clear sky shown in black
            bg = mask[t, 0].numpy() <= 0.1
            gt, pr = np.ma.masked_where(bg, gt), np.ma.masked_where(bg, pr)
        tag = ens.cfg["name"]
        save_panel(img, os.path.join(args.output_dir, f"{tag}_seq{args.sequence}_t{t}_input.pdf"))
        save_panel(gt, os.path.join(args.output_dir, f"{tag}_seq{args.sequence}_t{t}_gt.pdf"), cmap, norm, True)
        save_panel(pr, os.path.join(args.output_dir, f"{tag}_seq{args.sequence}_t{t}_pred.pdf"), cmap, norm, True)
        rows.append((label, img, gt, pr, norm))

    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.6 * len(rows)), dpi=200, squeeze=False,
                             constrained_layout=True)
    for r, (label, img, gt, pr, norm) in enumerate(rows):
        axes[r, 0].imshow(img, cmap="gray"); axes[r, 0].set_ylabel(label, fontsize=13, fontweight="bold")
        axes[r, 1].imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
        im = axes[r, 2].imshow(pr, cmap=cmap, norm=norm, interpolation="nearest")
        add_colorbar(fig, im, axes[r, :].tolist(), norm)
        for c, title in enumerate(["Input (satellite A)", "Ground truth $V_z$ [m/s]", "Inferred $V_z$ [m/s]"]):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(title, fontsize=12)
    out = os.path.join(args.output_dir, f"fig7_seq{args.sequence}_t{args.frame}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out} (+ per-panel PDFs)")


if __name__ == "__main__":
    main()
