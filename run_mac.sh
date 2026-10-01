#!/bin/bash
# One-command LAN launcher for the saw chamber on Apple Silicon.
#
#   ./run_mac.sh                 # Qwen3-4B @ layer 18 (repo default)
#   ./run_mac.sh 14b             # Qwen3-14B @ layer 22 (re-extracts vectors)
#   ./run_mac.sh 32b             # Qwen3-32B @ layer 32 (needs ~70 GB free)
#   CHAMBER_MODEL=... CHAMBER_LAYER=... ./run_mac.sh   # full manual override
#
# Serves BOTH the static site and the API from one uvicorn process:
#   http://<this-mac's-LAN-IP>:8000/          -> the site
#   http://<this-mac's-LAN-IP>:8000/live.html -> the live chamber UI
set -euo pipefail
cd "$(dirname "$0")"

# Let MPS use (almost) all of unified memory instead of the conservative
# recommendedMaxWorkingSet cap, and fall back to CPU for rare missing kernels.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
# 每次生成的最大新 token 数(默认 256),想更长的自述可以:
#   CHAMBER_MAX_NEW=400 ./run_mac.sh
# export CHAMBER_MAX_NEW=256

case "${1:-}" in
  14b) export CHAMBER_MODEL="${CHAMBER_MODEL:-Qwen/Qwen3-14B}"
       export CHAMBER_LAYER="${CHAMBER_LAYER:-22}" ;;
  32b) export CHAMBER_MODEL="${CHAMBER_MODEL:-Qwen/Qwen3-32B}"
       export CHAMBER_LAYER="${CHAMBER_LAYER:-32}" ;;
esac

PORT="${PORT:-8000}"
IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"

echo "model : ${CHAMBER_MODEL:-Qwen/Qwen3-4B} @ layer ${CHAMBER_LAYER:-18}"
echo "device: $(python3 -c 'import torch;print("mps" if torch.backends.mps.is_available() else "cpu")')"
echo
echo "  LAN:   http://${IP}:${PORT}/          (site)"
echo "  LAN:   http://${IP}:${PORT}/live.html (live chamber)"
echo "  local: http://localhost:${PORT}/live.html"
echo

exec uvicorn server:app --host 0.0.0.0 --port "$PORT" --app-dir live
