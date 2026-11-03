#!/usr/bin/env bash
# Train every experiment of the paper (4 folds each). ~1 h per fold on one RTX 3090.
#   DATA_ROOT=data OUT=checkpoints bash scripts/train_all.sh
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-data}
OUT=${OUT:-checkpoints}
EXPERIMENTS=(
  envelope layer_500m layer_1000m layer_1500m                                   # Table 1
  ablation_no_convlstm ablation_resnet18 ablation_resnet18_no_convlstm         # Table 2
  envelope_mono layer_500m_mono layer_1000m_mono layer_1500m_mono               # Table 3
  ablation_loss_l1 ablation_loss_power                                          # Fig. 6 (right)
  layer_900m_mono                                                               # SEAC4RS (Sec. 5)
)
for exp in "${EXPERIMENTS[@]}"; do
  python train.py --config "configs/${exp}.yaml" --data-root "$DATA_ROOT" --output-dir "$OUT"
done
