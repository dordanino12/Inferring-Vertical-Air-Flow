#!/usr/bin/env bash
# Evaluate the K-fold ensembles (Tables 1-3, Fig. 6).
#   DATA_ROOT=data CKPT=checkpoints bash scripts/eval_all.sh
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-data}
CKPT=${CKPT:-checkpoints}
for exp in envelope layer_500m layer_1000m layer_1500m \
           ablation_no_convlstm ablation_resnet18 ablation_resnet18_no_convlstm \
           envelope_mono layer_500m_mono layer_1000m_mono layer_1500m_mono \
           ablation_loss_l1 ablation_loss_power; do
  echo "=== ${exp}"
  python evaluate.py --config "configs/${exp}.yaml" --checkpoints "${CKPT}/${exp}" --data-root "$DATA_ROOT" --scatter
done
