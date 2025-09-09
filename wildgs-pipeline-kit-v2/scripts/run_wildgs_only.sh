#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 2 ] || { echo "Usage: $0 <WGS_DIR> <gpu_id> [config_rel]"; exit 1; }
WGS_DIR="$1"; GPU="$2"; CFG_REL="${3:-configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml}"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; fi
conda activate wildgs-slam || true
export CUDA_VISIBLE_DEVICES="$GPU"; export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128; export OMP_NUM_THREADS=8
cd "$WGS_DIR" || { echo "Bad WGS_DIR: $WGS_DIR"; exit 2; }
[ -f "$CFG_REL" ] || { echo "Config not found: $CFG_REL"; exit 3; }
python run.py "$CFG_REL"
