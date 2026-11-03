"""Compare a run's ensemble metrics (evaluate.py output) with the paper and the released checkpoints.

    python scripts/compare_with_paper.py --run runs/retrain --output runs/retrain/retrain_vs_paper.md
"""
import argparse
import json
import os

# Paper, Table 1: (MAE, ME, STD)
PAPER = {
    ("envelope", "masked"): (0.74, 0.21, 1.02),
    ("layer_500m", "unmasked"): (0.22, -0.07, 0.31), ("layer_500m", "masked"): (0.34, -0.03, 0.44),
    ("layer_1000m", "unmasked"): (0.19, 0.02, 0.38), ("layer_1000m", "masked"): (0.37, 0.06, 0.64),
    ("layer_1500m", "unmasked"): (0.16, 0.04, 0.32), ("layer_1500m", "masked"): (0.27, 0.07, 0.56),
}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/retrain")
    ap.add_argument("--released", default="results/eval", help="eval_metrics.json of the released checkpoints")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    rows = ["| Experiment | Pixels | MAE retrain / released / paper | ME retrain / released / paper "
            "| STD retrain / released / paper | max diff vs paper |", "|---|---|---|---|---|---|"]
    for (exp, pix), paper in PAPER.items():
        new = load(os.path.join(a.run, "eval", exp, "eval_metrics.json"))
        if not new or pix not in new:
            continue
        old = (load(os.path.join(a.released, exp, "eval_metrics.json")) or {}).get(pix, {})
        cells, diff = [], 0.0
        for i, k in enumerate(("MAE", "ME", "STD")):
            o = f"{old[k]:.3f}" if k in old else "–"
            cells.append(f"{new[pix][k]:.3f} / {o} / {paper[i]:.2f}")
            diff = max(diff, abs(new[pix][k] - paper[i]))
        rows.append(f"| `{exp}` | {pix} | " + " | ".join(cells) + f" | {diff:.3f} |")
    with open(a.output, "w") as f:
        f.write("# Retrained models vs. released checkpoints vs. paper (Table 1)\n\n"
                "4-fold ensemble on the shared test set (414 sequences), `evaluate.py`.\n\n" + "\n".join(rows) + "\n")
    print(f"wrote {a.output} ({len(rows) - 2} rows)")


if __name__ == "__main__":
    main()
