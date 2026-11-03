#!/usr/bin/env bash
# Retrain the Table 1 experiments (4 folds each, ~4 h per experiment on one RTX 3090),
# evaluate each ensemble, and compare with the paper and the released checkpoints.
# Safe to re-run: finished folds are skipped (--skip-existing).
#   tmux new -s retrain 'bash scripts/retrain_table1.sh'
#   OUT=runs/retrain EXPERIMENTS="envelope" bash scripts/retrain_table1.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-data}
OUT=${OUT:-runs/retrain}
EXPERIMENTS=${EXPERIMENTS:-"envelope layer_500m layer_1000m layer_1500m"}
mkdir -p "$OUT/logs" "$OUT/eval"
STATUS="$OUT/STATUS.txt"; HISTORY="$OUT/history.txt"
note() { echo "$(date '+%Y-%m-%d %H:%M') | $*" | tee -a "$HISTORY"; }

note "START experiments: $EXPERIMENTS (pid $$, host $(hostname))"
for exp in $EXPERIMENTS; do
  note "train $exp"
  if ! $PY -u train.py --config "configs/${exp}.yaml" --data-root "$DATA_ROOT" --output-dir "$OUT" \
       --skip-existing --status-file "$STATUS" >> "$OUT/logs/${exp}.log" 2>&1; then
    note "FAILED training $exp (see $OUT/logs/${exp}.log), continuing"; continue
  fi
  note "evaluate $exp"
  if $PY -u evaluate.py --config "configs/${exp}.yaml" --checkpoints "$OUT/${exp}" --data-root "$DATA_ROOT" \
       --scatter --output-dir "$OUT/eval/${exp}" >> "$OUT/logs/${exp}_eval.log" 2>&1; then
    note "done $exp: $(grep -E 'MAE=' "$OUT/logs/${exp}_eval.log" | tr -s ' ' | paste -sd ';' -)"
  else
    note "FAILED evaluating $exp (see $OUT/logs/${exp}_eval.log)"
  fi
  $PY scripts/compare_with_paper.py --run "$OUT" --output "$OUT/retrain_vs_paper.md" >> "$HISTORY" 2>&1 || true
done
echo "$(date '+%Y-%m-%d %H:%M') | ALL DONE - see $OUT/retrain_vs_paper.md" > "$STATUS"
note "ALL DONE"
