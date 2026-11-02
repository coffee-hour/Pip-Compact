#!/usr/bin/env bash
# Build the UltraChat dialog corpus at data/input.txt and wipe out/ if it changed.
#   - Installs the 'datasets' package if it is missing.
#   - Runs scripts/prep_dialog.py to stream HuggingFaceH4/ultrachat_200k.
#   - If the corpus content changed, deletes out/ so the tokenizer and checkpoints are rebuilt.
# Usage: bash download_chat_data.sh [prep_dialog.py flags, e.g. --max-conversations 5000]
# Then train with: bash train.sh
set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA:-data/input.txt}"

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "ERROR: Python 3 not found." >&2; exit 1
fi

if ! "$PY" -c "import datasets" >/dev/null 2>&1; then
  echo "Installing 'datasets' package..."
  "$PY" -m pip install -q datasets || { echo "ERROR: failed to install datasets." >&2; exit 1; }
fi

hash_file() {
  if [ ! -s "$1" ]; then echo none; return; fi
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}
BEFORE=$(hash_file "$DATA")

mkdir -p "$(dirname "$DATA")"
"$PY" scripts/prep_dialog.py --out "$DATA" "$@"

AFTER=$(hash_file "$DATA")
if [ "$BEFORE" != "$AFTER" ]; then
  if [ -d out ]; then
    echo "Corpus changed; removing out/ so the tokenizer and checkpoints are rebuilt."
    rm -rf out
  fi
else
  echo "Corpus unchanged; keeping out/."
fi

echo "Done. Train with: bash train.sh"
