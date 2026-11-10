#!/usr/bin/env bash
# Retrain experiments of the paper (4 folds each, ~4-5 h per experiment on one RTX 3090), evaluate
# each ensemble, and compare with the paper and the released checkpoints (OUT/retrain_vs_paper.md).
# layer_900m_mono is also run on the SEAC4RS AirMSPI data (needs AIRMSPI_NPZ and DC8_CSV).
# Safe to re-run: finished folds are skipped (--skip-existing). Run it detached, e.g.
#   tmux new -s retrain 'bash scripts/retrain.sh'              # every experiment (~3 days)
#   EXPERIMENTS="envelope layer_500m" bash scripts/retrain.sh  # a subset
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-data}
OUT=${OUT:-runs/retrain}
EXPERIMENTS=${EXPERIMENTS:-"envelope layer_500m layer_1000m layer_1500m
  envelope_mono layer_500m_mono layer_1000m_mono layer_1500m_mono layer_900m_mono"}
AIRMSPI_NPZ=${AIRMSPI_NPZ:-results/seac4rs/airmspi.npz}
DC8_CSV=${DC8_CSV:-results/seac4rs/dc8.csv}
mkdir -p "$OUT/logs" "$OUT/eval"
STATUS="$OUT/STATUS.txt"; HISTORY="$OUT/history.txt"
note() { echo "$(date '+%Y-%m-%d %H:%M') | $*" | tee -a "$HISTORY"; }

note "START experiments: $(echo $EXPERIMENTS) (pid $$, host $(hostname))"
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
  if [ "$exp" = layer_900m_mono ]; then
    note "SEAC4RS inference with $exp"
    if $PY -u real_data/seac4rs/infer_airmspi.py --npz "$AIRMSPI_NPZ" --checkpoints "$OUT/${exp}" \
         --kfold-dir "$DATA_ROOT/layer_900m" --output-dir "$OUT/seac4rs" > "$OUT/logs/seac4rs.log" 2>&1 &&
       $PY -u real_data/seac4rs/compare_distributions.py --insitu "$DC8_CSV" --model "$OUT/seac4rs/model_vz_pooled.csv" \
         --output "$OUT/seac4rs/fig11.pdf" >> "$OUT/logs/seac4rs.log" 2>&1; then
      note "SEAC4RS: $(grep Wasserstein "$OUT/logs/seac4rs.log")"
    else
      note "FAILED SEAC4RS (see $OUT/logs/seac4rs.log)"
    fi
  fi
  $PY scripts/compare_with_paper.py --run "$OUT" --output "$OUT/retrain_vs_paper.md" >> "$HISTORY" 2>&1 || true
done
echo "$(date '+%Y-%m-%d %H:%M') | ALL DONE - see $OUT/retrain_vs_paper.md" > "$STATUS"
note "ALL DONE"
