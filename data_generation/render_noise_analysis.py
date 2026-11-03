"""
Supplementary: Monte Carlo rendering noise vs. photon (Poisson) noise.

Renders one cloud sample from one satellite view N times with different seeds,
converts radiance to electrons (the brightest pixel fills the 10,600 e- well) and
plots the per-pixel variance across renders against the mean. Points near or above
the Poisson line (variance = mean) show that path-tracing noise is already at least
as large as photon noise, which is why no photon noise is added to the dataset.

    python data_generation/render_noise_analysis.py --sample volumes/0000005920/sample_012.pkl --output results/render_noise.pdf
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mitsuba_renderer import MitsubaRenderer  # noqa: E402

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "overpass_2satellites.csv")
FULL_WELL = 10600          # [e-]
C_EFF = 8.73e5             # radiance -> electron-rate constant
DARK_CURRENT = 4.72        # [e-/s]
READ_NOISE_VAR = 5.29      # [e-^2]
QUANT_VAR = 8.93           # [e-^2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, help="a volume pickle from les_to_volumes.py")
    ap.add_argument("--overpass-csv", default=DEFAULT_CSV)
    ap.add_argument("--row", type=int, default=9, help="overpass CSV row (one satellite at one time)")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="seeds seed..seed+runs-1")
    ap.add_argument("--spp", type=int, default=8192)
    ap.add_argument("--g", type=float, default=0.85)
    ap.add_argument("--ocean-reflectance", type=float, default=0.03, help="negative = no ocean")
    ap.add_argument("--camera-model", action="store_true", help="also draw the full camera noise model")
    ap.add_argument("--output", default="results/render_noise.pdf")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    vol = os.path.splitext(args.output)[0] + "_tmp.vol"
    imgs = []
    for k in tqdm(range(args.runs), desc="renders"):
        r = MitsubaRenderer(args.overpass_csv, [args.row], spp=args.spp, g_value=args.g, vol_path=vol,
                            seed=args.seed + k,
                            ocean_reflectance=None if args.ocean_reflectance < 0 else args.ocean_reflectance)
        imgs.append(r.render(args.sample)[0])
    os.remove(vol)
    stack = np.stack(imgs)

    delta_t = FULL_WELL / (stack.max() * C_EFF)                 # exposure filling the well
    e = stack * C_EFF * delta_t
    mean, var = e.mean(axis=0), e.var(axis=0)
    sel = mean > 1
    m, v = mean[sel], var[sel]
    bright = m > np.percentile(m, 90)          # cloud pixels (the ocean background is nearly noise-free)
    print(f"exposure {delta_t * 1e3:.3f} ms, {sel.sum()} pixels; brightest 10%: "
          f"median MC variance / Poisson variance = {np.median(v[bright] / m[bright]):.2f}")

    order = np.argsort(m)
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(m, v, s=1, alpha=0.3, color="gray", label="Monte Carlo Variance (Render)", rasterized=True)
    plt.plot(m[order], m[order], "r-", lw=2, label="Poisson Noise Model (Var = Mean)")
    if args.camera_model:
        plt.plot(m[order], m[order] + DARK_CURRENT * delta_t + READ_NOISE_VAR + QUANT_VAR, "b--", lw=2,
                 label="Full Camera Noise Model")
    plt.title("Noise Comparison", fontsize=20)
    plt.xlabel("Mean Signal (Electrons - $u_e$)", fontsize=18); plt.ylabel(r"Variance ($\sigma^2$)", fontsize=18)
    plt.legend(fontsize=14); plt.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    np.savez(os.path.splitext(args.output)[0] + "_stack.npz", radiance=stack)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
