#!/usr/bin/env bash
# Pip-Compact one-command launcher.
#   - Installs requirements.txt if torch or numpy is missing (uses an active venv if one is set).
#   - If data/input.txt is missing, runs download_chat_data.sh (curated UltraChat prep) automatically.
#     If that fails, falls back to downloading TinyShakespeare so training can still start.
#   - Uses CUDA when present (fp16 autocast); otherwise fp32 on CPU or MPS.
#   - Auto-resumes from out/ckpt.pt when it exists.
# Usage: bash train.sh [STEPS] [train.py flags, e.g. --steps 8000 --n_layer 12]
#   bash train.sh 20000            same as --steps 20000 (total target step count)
#   STEPS=20000 bash train.sh      same, via environment variable
#   If the checkpoint is at step 5000 and the target is 20000, training continues from 5000 to 20000.
set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA:-data/input.txt}"
DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
mkdir -p "$(dirname "$DATA")" out

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then PY="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "ERROR: Python 3 not found. Install Python 3 or set PYTHON=/path/to/python." >&2; exit 1
fi

if ! "$PY" -c "import torch, numpy" >/dev/null 2>&1; then
  echo "Installing dependencies from requirements.txt..."
  if ! "$PY" -m pip install -q -r requirements.txt; then
    echo "ERROR: pip install failed. If your Python is externally managed, create a venv first:" >&2
    echo "  python3 -m venv .venv && . .venv/bin/activate && bash train.sh" >&2
    exit 1
  fi
fi

if [ ! -s "$DATA" ]; then
  echo "No corpus at $DATA. Running dataset prep (download_chat_data.sh)..."
  if PYTHON="$PY" DATA="$DATA" bash download_chat_data.sh; then
    echo "Dataset prep finished."
  else
    echo "WARNING: dataset prep failed. Falling back to TinyShakespeare." >&2
    curl -fsSL -o "$DATA" "$DATA_URL" || { echo "ERROR: could not download a corpus to $DATA." >&2; exit 1; }
  fi
fi

if [ ! -s "$DATA" ]; then
  echo "ERROR: corpus still missing at $DATA after dataset prep." >&2
  exit 1
fi

DEVICE=$("$PY" -c "import torch; m = getattr(torch.backends, 'mps', None); print('cuda' if torch.cuda.is_available() else ('mps' if m and m.is_available() else 'cpu'))")
echo "device: $DEVICE"

# A bare leading integer is the total step target: `bash train.sh 20000` -> `--steps 20000`.
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  set -- --steps "$1" "${@:2}"
fi
USER_ARGS=("$@")
has_flag() { local f; for f in "${USER_ARGS[@]+"${USER_ARGS[@]}"}"; do [ "$f" = "$1" ] && return 0; done; return 1; }

# STEPS env var is used only when no --steps flag was given.
if [ -n "${STEPS:-}" ] && ! has_flag --steps; then
  USER_ARGS+=(--steps "$STEPS")
fi

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
