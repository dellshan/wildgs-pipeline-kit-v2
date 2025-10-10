#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "[i] Using Python: $(python -c 'import sys; print(sys.executable)')"

# Install deps into the *current* env once if needed:
if [[ "${INSTALL:-0}" == "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

export MPLBACKEND=Agg
python scripts/make_figs.py --config "${1:-config.json}"
echo "[OK] figs/ 和 tables/ 已更新"
