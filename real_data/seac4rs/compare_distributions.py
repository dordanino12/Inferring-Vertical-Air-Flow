"""
Step 4 - compare inferred and in-situ V_z distributions (Fig. 11).

Reports value ranges, the Wasserstein (earth mover's) distance and the MAE
between matched quantiles (1st..99th percentile), and draws the density
overlay with the Q-Q points on a twin axis.

    python real_data/seac4rs/compare_distributions.py --insitu dc8_window.csv \
        --model results/seac4rs/model_vz_pooled.csv --output results/seac4rs/fig11.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402


def load_column(path, column):
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    return df[column].dropna().values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insitu", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="fig11_seac4rs.pdf")
    args = ap.parse_args()

    w_in = load_column(args.insitu, "Vertical Velocity (m/s)")
    w_mo = load_column(args.model, "Model Vertical Velocity (m/s)")
    pct = np.linspace(1, 99, 100)
    q_in, q_mo = np.percentile(w_in, pct), np.percentile(w_mo, pct)
    emd = wasserstein_distance(w_in, w_mo)
    qq_mae = float(np.mean(np.abs(q_in - q_mo)))
    print(f"DC-8  : N={w_in.size}  range [{w_in.min():.2f}, {w_in.max():.2f}]  mean {w_in.mean():.2f}  std {w_in.std():.2f} m/s")
    print(f"Model : N={w_mo.size}  range [{w_mo.min():.2f}, {w_mo.max():.2f}]  mean {w_mo.mean():.2f}  std {w_mo.std():.2f} m/s")
    print(f"Wasserstein distance {emd:.3f} m/s | Q-Q MAE {qq_mae:.3f} m/s")

    c_in, c_mo, c_qq = "#348ABD", "#E24A33", "forestgreen"
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    sns.kdeplot(w_in, color=c_in, fill=True, alpha=0.4, linewidth=0, bw_adjust=1.2, ax=ax)
    sns.kdeplot(w_in, color="#1D4E89", linewidth=2, linestyle="--", bw_adjust=1.2, ax=ax)
    sns.kdeplot(w_mo, color=c_mo, fill=True, alpha=0.4, linewidth=0, bw_adjust=1.0, ax=ax)
    sns.kdeplot(w_mo, color="#9E2A2B", linewidth=2, bw_adjust=1.0, ax=ax)
    lo, hi = min(w_in.min(), w_mo.min()), max(w_in.max(), w_mo.max())
    ax.set_xlim([lo - 3, hi + 2]); ax.set_ylim([0, 0.8]); ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_xlabel("Vertical Velocity [m/s]"); ax.set_ylabel("Probability Density"); ax.grid(True, linestyle="--", alpha=0.6)
    ax2 = ax.twinx()
    xl = ax.get_xlim()
    ax2.scatter(q_in, q_mo, color=c_qq, alpha=0.7, edgecolor="k", zorder=3)
    ax2.plot(xl, xl, color=c_qq, linestyle="--", linewidth=2)
    ax2.set_ylim(xl); ax2.set_ylabel("Q-Q Model Inference [m/s]", color=c_qq)
    ax2.tick_params(axis="y", colors=c_qq); ax2.spines["right"].set_color(c_qq)
    ax.legend(handles=[mpatches.Patch(color=c_in, alpha=0.4, label="Aircraft\nMeasurements\n (DC-8)"),
                       mpatches.Patch(color=c_mo, alpha=0.4, label="Model (Ours)"),
                       mlines.Line2D([], [], color=c_qq, marker="o", linestyle="None", markeredgecolor="k", label="Q-Q Percentiles"),
                       mlines.Line2D([], [], color=c_qq, linestyle="--", label="1:1 Ideal Match")],
              loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(args.output, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
