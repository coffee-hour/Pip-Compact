#!/usr/bin/env bash
# Pip-Compact launcher. Kaggle and Colab already ship PyTorch and CUDA.
# Usage: bash train.sh [--fp16|--bf16] [--steps N] [--batch N] [--accum N] [--resume] ...
set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA:-data/input.txt}"
mkdir -p data out

if [ ! -f "$DATA" ]; then
  echo "No corpus at $DATA. Downloading TinyShakespeare as a default corpus..."
  curl -fsSL -o "$DATA" https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo 'No CUDA GPU detected. In Kaggle or Colab, set the accelerator to GPU T4.'; exit 1; }
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

has_prec=0
for arg in "$@"; do
  case "$arg" in --fp16|--bf16) has_prec=1 ;; esac
done
ARGS=("$@")
if [ "$has_prec" -eq 0 ]; then ARGS+=("--fp16"); fi

python train.py --data "$DATA" "${ARGS[@]}"
