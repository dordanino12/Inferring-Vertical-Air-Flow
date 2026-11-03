"""
Supplementary video: satellite geometry + rendered stereo images, then ground-truth
vs. inferred V_z for the cloud envelope and three horizontal layers, for one test sequence.

    python figures/demo_video.py --sequence 200 --output results/supplementary_video.mp4 \
        --envelope configs/envelope.yaml:checkpoints/envelope \
        --layers configs/layer_500m.yaml:checkpoints/layer_500m \
                 configs/layer_1000m.yaml:checkpoints/layer_1000m \
                 configs/layer_1500m.yaml:checkpoints/layer_1500m

By default only the first fold's model of each target is used (as in the published
video); --all-folds uses the K-fold ensemble.
"""
import argparse
import ast
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from vertical_airflow.ensemble import Ensemble  # noqa: E402
from vertical_airflow.utils import fold_dirs  # noqa: E402

DEFAULT_CSV = os.path.join(ROOT, "assets", "overpass_2satellites.csv")
TITLE = "Inferring Vertical Air Flow by Spaceborne Imaging"
LINTHRESH = 3.0
FIG_SIZE, DPI = (18, 8), 120


def load_positions(csv_path):
    """Satellite positions [m] per acquisition time (same frame as the ray caster)."""
    df = pd.read_csv(csv_path)
    times = sorted(df["utc time"].unique())
    pos = {t: [np.array([-e[1], e[0], e[2]]) * 1000.0 for e in
               (ast.literal_eval(s) for s in df[df["utc time"] == t]["sat ENU coordinates [km]"])] for t in times}
    return times, pos


def fig_to_bgr(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba[..., :3].copy(), cv2.COLOR_RGB2BGR)


def gamma(img, g=0.5):
    lo, hi = float(img.min()), float(img.max())
    return np.zeros_like(img) if hi - lo < 1e-8 else ((img - lo) / (hi - lo)) ** g


def geometry_axes(ax, sat_positions, bounds, look_at=(0, 0, 1500)):
    cx, cy, cz = (v / 1000 for v in look_at)
    ax.scatter(cx, cy, cz, c="#555555", s=320, marker="X", depthshade=False)
    for i, (p, color, name) in enumerate(zip(sat_positions, ["#E74C3C", "#3498DB"], ["SAT A", "SAT B"])):
        x, y, z = p / 1000
        ax.scatter(x, y, z, c=color, s=230, depthshade=False, edgecolors="white", linewidth=3.5)
        ax.plot([x, cx], [y, cy], [z, cz], c=color, linestyle="--", linewidth=4)
        ax.text(x + (-1 if i == 0 else 1) * max(bounds[0] * 0.03, 3), y, z + bounds[2] * 0.05, name,
                color=color, fontsize=14, fontweight="bold", ha="right" if i == 0 else "left")
    ax.set_xlim(-bounds[0], bounds[0]); ax.set_ylim(-bounds[1], bounds[1]); ax.set_zlim(0, bounds[2])
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
    ax.view_init(elev=25, azim=-45)


def colorbar(fig, im, ax, vmin, vmax):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.array([vmin, vmin / 2, 0.0, vmax / 2, vmax]) if vmin < 0 < vmax else np.linspace(vmin, vmax, 5)
    cb.set_ticks(ticks); cb.ax.set_ylim(vmin, vmax); cb.ax.tick_params(labelsize=12)
    cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envelope", required=True, help="CONFIG:CHECKPOINT_DIR")
    ap.add_argument("--layers", nargs="+", required=True, help="CONFIG:CHECKPOINT_DIR per layer")
    ap.add_argument("--sequence", type=int, default=200)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--overpass-csv", default=DEFAULT_CSV)
    ap.add_argument("--output", default="results/supplementary_video.mp4")
    ap.add_argument("--fps", type=float, default=0.7)
    ap.add_argument("--title-seconds", type=float, default=5)
    ap.add_argument("--all-folds", action="store_true")
    args = ap.parse_args()

    panels = []   # (label, gt, pred, mask or None, vmin, vmax)
    stereo = None
    for i, spec in enumerate([args.envelope] + args.layers):
        cfg, ckpt = spec.split(":")[:2]
        probe = Ensemble(cfg, ckpt, args.data_root, verbose=False) if args.all_folds else None
        if probe is None:
            from vertical_airflow.utils import load_config
            root = os.path.join(args.data_root, load_config(cfg)["data"]["kfold_dir"])
            probe = Ensemble(cfg, ckpt, args.data_root, folds=[fold_dirs(root)[0]], verbose=False)
        ds = probe.test_set()
        x, y, mask = ds[args.sequence]
        pred, _ = probe.predict(x[None])
        gt = ds.denormalize(y.numpy())
        label = "Envelope" if i == 0 else f"{probe.cfg['name'].split('_')[1]}"
        panels.append((label, gt, pred[0], mask.numpy() if i == 0 else None, float(ds.Y.min()), float(ds.Y.max())))
        if stereo is None:
            stereo = x.numpy()
        print(f"[video] {label}: {len(probe.models)} model(s)")

    times, positions = load_positions(args.overpass_csv)
    allp = np.array([p for t in times for p in positions[t]]) / 1000
    bounds = (np.abs(allp[:, 0]).max() * 1.2, np.abs(allp[:, 1]).max() * 1.2, allp[:, 2].max() * 1.1)
    cmap = mcolors.ListedColormap(plt.get_cmap("jet", 256)(np.linspace(0, 1, 256)))
    cmap.set_bad("black")
    T = stereo.shape[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    size = fig_to_bgr(plt.figure(figsize=FIG_SIZE, dpi=DPI)).shape[1::-1]
    plt.close("all")
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)

    def write(fig):
        frame = fig_to_bgr(fig); plt.close(fig)
        if frame.shape[1::-1] != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(frame)

    for _ in range(max(1, int(args.fps * args.title_seconds))):          # part 1: title
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI); fig.patch.set_facecolor("#f5f5f5")
        fig.text(0.5, 0.55, TITLE, ha="center", va="center", fontsize=32, fontweight="bold")
        write(fig)

    for t in range(T):                                                     # part 2: geometry + images
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
        gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
        fig.suptitle(f"Geometry and Images | Time: {t * 20} sec", fontsize=20, fontweight="bold")
        geometry_axes(fig.add_subplot(gs[:, 0], projection="3d"), positions[times[t % len(times)]], bounds)
        for r, name in enumerate(["Image at Sat A", "Image at Sat B"]):
            a = fig.add_subplot(gs[r, 1]); a.imshow(gamma(stereo[t, r]), cmap="gray", vmin=0, vmax=1)
            a.set_title(name, fontsize=16, fontweight="bold"); a.axis("off")
        write(fig)

    for t in range(T):                                                     # part 3: GT vs inferred
        fig, axes = plt.subplots(2, len(panels), figsize=FIG_SIZE, dpi=DPI, squeeze=False)
        fig.suptitle(f"GT vs Inferred Vertical Velocity | Time: {t * 20} sec", fontsize=20, fontweight="bold")
        for c, (label, gt, pred, mask, vmin, vmax) in enumerate(panels):
            norm = mcolors.SymLogNorm(linthresh=LINTHRESH, linscale=1.0, vmin=vmin, vmax=vmax)
            g, p = gt[t, 0], pred[t, 0]
            if mask is not None:
                g, p = np.ma.masked_where(mask[t, 0] == 0, g), np.ma.masked_where(mask[t, 0] == 0, p)
            for r, (img, kind) in enumerate([(g, "GT"), (p, "Inferred")]):
                im = axes[r, c].imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
                axes[r, c].set_title(f"{label} {kind} [m/s]", fontsize=14, fontweight="bold"); axes[r, c].axis("off")
                colorbar(fig, im, axes[r, c], vmin, vmax)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        write(fig)

    writer.release()
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
