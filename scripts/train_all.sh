#!/usr/bin/env bash
# Train every experiment of the paper (4 folds each). ~1 h per fold on one RTX 3090.
#   DATA_ROOT=data OUT=checkpoints bash scripts/train_all.sh
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-data}
OUT=${OUT:-checkpoints}
EXPERIMENTS=(
  envelope layer_500m layer_1000m layer_1500m                                   # Table 1
  envelope_mono layer_500m_mono layer_1000m_mono layer_1500m_mono               # Table 3
  layer_900m_mono                                                               # SEAC4RS (Sec. 5)
)
for exp in "${EXPERIMENTS[@]}"; do
  python train.py --config "configs/${exp}.yaml" --data-root "$DATA_ROOT" --output-dir "$OUT"
done
