#!/usr/bin/env bash
# Upload or download Pip-Compact weights via GitHub Releases, keeping them out of git history.
# Usage:
#   bash scripts/upload_weights.sh upload [tag]     # default tag v0.1
#   bash scripts/upload_weights.sh download [tag]   # fetches ckpt.pt and tokenizer.json into out/
# Env overrides: REPO (default coffee-hour/Pip-Compact), CKPT (out/ckpt.pt), TOK (out/tokenizer.json)
set -euo pipefail

REPO="${REPO:-coffee-hour/Pip-Compact}"
CKPT="${CKPT:-out/ckpt.pt}"
TOK="${TOK:-out/tokenizer.json}"
MODE="${1:-}"
TAG="${2:-v0.1}"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required: https://cli.github.com  then run: gh auth login" >&2
  exit 1
fi

case "$MODE" in
  upload)
    [ -f "$CKPT" ] || { echo "missing $CKPT (train first with bash train.sh)" >&2; exit 1; }
    ASSETS=("$CKPT")
    [ -f "$TOK" ] && ASSETS+=("$TOK")
    if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
      echo "release $TAG exists, replacing assets"
      gh release upload "$TAG" "${ASSETS[@]}" --repo "$REPO" --clobber
    else
      gh release create "$TAG" "${ASSETS[@]}" --repo "$REPO" \
        --title "Pip-Compact $TAG" \
        --notes "Trained Pip-Compact weights (out/ckpt.pt) and BPE tokenizer (out/tokenizer.json)."
    fi
    ;;
  download)
    mkdir -p out
    gh release download "$TAG" --repo "$REPO" --pattern "ckpt.pt" --pattern "tokenizer.json" --dir out --clobber
    echo "downloaded to out/"
    ;;
  *)
    echo "usage: bash scripts/upload_weights.sh {upload|download} [tag]" >&2
    exit 1
    ;;
esac
