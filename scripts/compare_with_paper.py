"""Compare a run's ensemble metrics (evaluate.py output) with the paper and the released checkpoints.

    python scripts/compare_with_paper.py --run runs/retrain --output runs/retrain/retrain_vs_paper.md
"""
import argparse
import json
import os
import re

# (experiment, pixels): (paper table, MAE, ME, STD); None = not reported in the paper
PAPER = {
    ("envelope", "masked"): ("Table 1", 0.74, 0.21, 1.02),
    ("layer_500m", "unmasked"): ("Table 1", 0.22, -0.07, 0.31),
    ("layer_500m", "masked"): ("Table 1", 0.34, -0.03, 0.44),
    ("layer_1000m", "unmasked"): ("Table 1", 0.19, 0.02, 0.38),
    ("layer_1000m", "masked"): ("Table 1", 0.37, 0.06, 0.64),
    ("layer_1500m", "unmasked"): ("Table 1", 0.16, 0.04, 0.32),
    ("layer_1500m", "masked"): ("Table 1", 0.27, 0.07, 0.56),
    ("envelope_mono", "masked"): ("Table 3", 0.81, None, 1.04),
    ("layer_500m_mono", "unmasked"): ("Table 3", 0.23, None, 0.34),
    ("layer_500m_mono", "masked"): ("Table 3", 0.37, None, 0.48),
    ("layer_1000m_mono", "unmasked"): ("Table 3", 0.21, None, 0.45),
    ("layer_1000m_mono", "masked"): ("Table 3", 0.41, None, 0.68),
    ("layer_1500m_mono", "unmasked"): ("Table 3", 0.19, None, 0.41),
    ("layer_1500m_mono", "masked"): ("Table 3", 0.33, None, 0.67),
}
SEAC4RS_PAPER = {"Wasserstein": 0.15, "Q-Q MAE": 0.14}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fmt(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/retrain")
    ap.add_argument("--released", default="results/eval", help="eval_metrics.json of the released checkpoints")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    rows = ["| Paper | Experiment | Pixels | MAE retrain / released / paper | ME retrain / released / paper "
            "| STD retrain / released / paper | max diff vs paper |", "|---|---|---|---|---|---|---|"]
    for (exp, pix), (table, *paper) in PAPER.items():
        new = load(os.path.join(a.run, "eval", exp, "eval_metrics.json"))
        if not new or pix not in new:
            continue
        old = (load(os.path.join(a.released, exp, "eval_metrics.json")) or {}).get(pix, {})
        cells, diffs = [], []
        for k, p in zip(("MAE", "ME", "STD"), paper):
            cells.append(f"{new[pix][k]:.3f} / {fmt(old.get(k))} / {fmt(p, 2)}")
            if p is not None:
                diffs.append(abs(new[pix][k] - p))
        rows.append(f"| {table} | `{exp}` | {pix} | " + " | ".join(cells) + f" | {fmt(max(diffs) if diffs else None)} |")
    text = ("# Retrained models vs. released checkpoints vs. paper\n\n"
            "4-fold ensemble on the shared test set (414 sequences), `evaluate.py`. "
            "Envelope: cloudy pixels; layers: unmasked = all pixels, masked = dilated cloud region.\n\n" + "\n".join(rows) + "\n")
    log = os.path.join(a.run, "logs", "seac4rs.log")
    if os.path.exists(log):
        s = open(log).read()
        w = re.search(r"Wasserstein distance\s*([0-9.]+)", s)
        q = re.search(r"Q-Q MAE\s*([0-9.]+)", s)
        if w and q:
            text += ("\n## SEAC4RS (layer_900m_mono, Fig. 11)\n\n| | Retrain | Released | Paper |\n|---|---|---|---|\n"
                     f"| Wasserstein distance [m/s] | {w.group(1)} | 0.146 | {SEAC4RS_PAPER['Wasserstein']} |\n"
                     f"| Q-Q MAE [m/s] | {q.group(1)} | 0.143 | {SEAC4RS_PAPER['Q-Q MAE']} |\n")
    with open(a.output, "w") as f:
        f.write(text)
    print(f"wrote {a.output} ({len(rows) - 2} rows)")


if __name__ == "__main__":
    main()
