#!/usr/bin/env bash
set -euo pipefail

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
  # 默认跑 bicycle
  SCENES=(bicycle)
fi

for S in "${SCENES[@]}"; do
  echo ">>> [${S}] prepare"
  python tools/prepare_mipnerf360_sample.py "${S}" \
    --root data/mipnerf360/raw \
    --out_root outputs/mipnerf360 \
    --num 60

  echo ">>> [${S}] yolo"
  python tools/yolo_detect_dir.py \
    --images outputs/mipnerf360/${S}/images \
    --out    outputs/mipnerf360/${S}

  echo ">>> [${S}] to csv"
  python tools/boxes_to_csv.py \
    --images outputs/mipnerf360/${S}/images \
    --labels outputs/mipnerf360/${S}/labels \
    --out_csv outputs/mipnerf360/${S}/yolo_preds.csv
done

echo "[DONE] mipnerf360 scenes: ${SCENES[*]}"
