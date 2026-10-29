#!/usr/bin/env bash
# Pip-Compact one-command launcher.
#   - Installs torch if it is missing.
#   - Downloads TinyShakespeare to data/input.txt if the corpus is missing.
#   - Uses CUDA when present (fp16 autocast); otherwise fp32 on CPU or MPS.
#   - Auto-resumes from out/ckpt.pt when it exists.
# Usage: bash train.sh [train.py flags, e.g. --steps 8000 --n_layer 12]
set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA:-data/input.txt}"
DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
mkdir -p "$(dirname "$DATA")" out

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "ERROR: Python 3 not found." >&2; exit 1
fi

if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "Installing PyTorch..."
  "$PY" -m pip install -q torch
fi

if [ ! -s "$DATA" ]; then
  echo "No corpus at $DATA. Downloading TinyShakespeare..."
  curl -fsSL -o "$DATA" "$DATA_URL"
fi

DEVICE=$("$PY" -c "import torch; m = getattr(torch.backends, 'mps', None); print('cuda' if torch.cuda.is_available() else ('mps' if m and m.is_available() else 'cpu'))")
echo "device: $DEVICE"

USER_ARGS=("$@")
has_flag() { local f; for f in "${USER_ARGS[@]+"${USER_ARGS[@]}"}"; do [ "$f" = "$1" ] && return 0; done; return 1; }

ARGS=()
if [ "$DEVICE" = "cuda" ]; then
  if ! has_flag --fp16 && ! has_flag --bf16; then ARGS+=(--fp16); fi
else
  for f in --fp16 --bf16; do
    if has_flag "$f"; then echo "note: $f needs CUDA; running fp32 on $DEVICE instead."; fi
  done
fi

if [ -f out/ckpt.pt ] && ! has_flag --resume; then
  echo "resuming from out/ckpt.pt"
  ARGS+=(--resume)
fi

exec "$PY" train.py --data "$DATA" ${ARGS[@]+"${ARGS[@]}"} ${USER_ARGS[@]+"${USER_ARGS[@]}"}
