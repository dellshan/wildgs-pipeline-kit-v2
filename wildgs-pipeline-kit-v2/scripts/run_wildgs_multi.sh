#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 3 ] || { echo "Usage: $0 <WGS_DIR> <gpu_list> <config1> [config2 ...]"; exit 1; }
WGS_DIR="$1"; shift; GPU_LIST="$1"; shift
IFS=',' read -r -a GPUS <<< "$GPU_LIST"; NGPU=${#GPUS[@]}; [ "$NGPU" -gt 0 ] || { echo "No GPUs parsed"; exit 2; }
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; fi; conda activate wildgs-slam || true
cd "$WGS_DIR"
i=0; for cfg in "$@"; do g="${GPUS[$((i % NGPU))]}"; log="wildgs_gpu${g}_$(basename "$cfg" .yaml).log"
  echo "[launch] GPU=$g CONFIG=$cfg -> $log"
  CUDA_VISIBLE_DEVICES="$g" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 OMP_NUM_THREADS=8     python run.py "$cfg" > "$log" 2>&1 &
  i=$((i+1))
done
echo "Launched $i runs. tail -f wildgs_*.log"
