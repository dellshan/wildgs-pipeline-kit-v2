#!/usr/bin/env bash
set -euo pipefail
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=(bicycle)

# 若你已有 SAM2 批处理命令，设置环境变量 SAM2_CMD 触发 SAM2 模式
# 例：
# export SAM2_CMD='python your_sam2_batch.py --images "{IMG}" --yolo_labels "{LAB}" --out "{MSK}"'

for S in "${SCENES[@]}"; do
  IMG="outputs/mipnerf360/${S}/images"
  LAB="outputs/mipnerf360/${S}/labels"
  OUT="outputs/mipnerf360/${S}"

  mkdir -p "${OUT}/masks"

  if [ -n "${SAM2_CMD:-}" ]; then
    echo ">>> [${S}] SAM2 (from YOLO prompts)"
    eval $(echo "$SAM2_CMD" | sed \
      -e "s|{IMG}|${IMG}|g" \
      -e "s|{LAB}|${LAB}|g" \
      -e "s|{MSK}|${OUT}/masks|g")
    # 你自己的 SAM2 脚本写出 per-instance 掩码到 ${OUT}/masks，并生成 ${OUT}/instances.csv（同下格式）
  else
    echo ">>> [${S}] box -> rectangle masks (fallback)"
    python tools/box_to_masks.py \
      --images "${IMG}" \
      --labels "${LAB}" \
      --out "${OUT}" \
      --min_conf 0.20
  fi

  echo ">>> [${S}] CLIP naming"
  python tools/clip_zero_shot_label.py \
    --images_root "${IMG}" \
    --instances_csv "${OUT}/instances.csv" \
    --out_csv "${OUT}/objects_views.csv"
done

echo "[DONE] stage2 for: ${SCENES[*]}"
