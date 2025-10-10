#!/usr/bin/env bash
# tools/run_pacificvis_all.sh
# 一键：生成 links → 重算障碍 → 出全套图表 → 论文拼图 → 打包可分享数据

set -euo pipefail

# ==== 可按需改动的配置 ====
ROOT="output/view_story_B_2d"                       # story / links / obstacles 的目录
STORY_JSON="$ROOT/story.with_names.en.json"         # 如果拿到新的标注，替换成它的路径即可
MASK_ROOT="output/instances_B_masks_conf08"         # 如无 mask，可留空 ""（仍可运行，只是面积用近似）
USE_POSES=0                                         # 1=用位姿；0=常量步长0.5m + yaw=0
POSES_JSON="output/poses_est_B.json"                # 仅当 USE_POSES=1 且你有这个文件时有效

OUT_FIGS_BASE="output/pacificvis_figs"
OUT_FIGS_PLUS="output/pacificvis_plus"
OUT_PLATE_BASE="output/pacificvis_plate"
OUT_ZIP_DIR="output/vis_sources_en"

# ==== 开始 ====
echo "[1/6] 准备 links.json ..."
if [[ "$USE_POSES" == "1" ]]; then
  python pipeline/poses_json_to_views_csv.py \
    --poses "$POSES_JSON" \
    --out_csv frames_views_posed.csv
  python pipeline/make_links_from_views_csv.py \
    --csv frames_views_posed.csv \
    --story "$STORY_JSON" \
    --out   "$ROOT/links.json"
else
  python pipeline/make_links_from_views_csv.py \
    --csv frames_views.csv \
    --story "$STORY_JSON" \
    --const_step 0.5 \
    --out   "$ROOT/links.json"
fi

echo "[2/6] 重算障碍风险 obstacles_recomputed.json ..."
python pipeline/recompute_obstacles.py \
  --story     "$STORY_JSON" \
  --mask_root "${MASK_ROOT:-}" \
  --out_dir   "$ROOT"

echo "[3/6] 生成基础图 (viz_pacific.py) ..."
python pipeline/viz_pacific.py \
  --story     "$STORY_JSON" \
  --links     "$ROOT/links.json" \
  --obstacles "$ROOT/obstacles_recomputed.json" \
  --out_dir   "$OUT_FIGS_BASE"

echo "[4/6] 生成加强图+表 (viz_pacific_plus.py) ..."
python pipeline/viz_pacific_plus.py \
  --story     "$STORY_JSON" \
  --links     "$ROOT/links.json" \
  --obstacles "$ROOT/obstacles_recomputed.json" \
  --out_dir   "$OUT_FIGS_PLUS"

echo "[5/6] 拼论文面板 (make_pacificvis_plate.py) ..."
python pipeline/make_pacificvis_plate.py \
  --fig_dir   "$OUT_FIGS_PLUS" \
  --story_dir "$ROOT" \
  --out       "$OUT_PLATE_BASE"

echo "[6/6] 打包可分享数据 (export_vis_sources.py) ..."
python pipeline/export_vis_sources.py \
  --story     "$STORY_JSON" \
  --links     "$ROOT/links.json" \
  --obstacles "$ROOT/obstacles_recomputed.json" \
  --extra     frames_views.csv frames_views_posed.csv \
  --outdir    "$OUT_ZIP_DIR"

# 小结 & 位置提示
echo
echo "================ DONE ================"
echo "基础图目录:         $OUT_FIGS_BASE"
echo "加强图+表目录:      $OUT_FIGS_PLUS"
echo "论文拼图:           ${OUT_PLATE_BASE}.png | ${OUT_PLATE_BASE}.pdf"
echo "可分享数据 ZIP:     ${OUT_ZIP_DIR}.zip"
if command -v sha256sum >/dev/null 2>&1; then
  echo -n "ZIP 校验和:        "; sha256sum "${OUT_ZIP_DIR}.zip" | awk '{print $1}'
fi
echo "======================================"
