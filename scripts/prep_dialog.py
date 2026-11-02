#!/usr/bin/env python3
"""Stream HuggingFaceH4/ultrachat_200k and write User/Assistant dialog to data/input.txt.

Usage: python scripts/prep_dialog.py [--out data/input.txt] [--dataset HuggingFaceH4/ultrachat_200k]
         [--split train_sft] [--max-conversations 20000] [--min-chars 0]

Each conversation becomes:
User: <message>
Assistant: <reply>

Conversations are separated by a blank line. The file is written atomically
(temp file, then rename), so a failed run never leaves a half-written corpus.
"""
import argparse
import os
import sys


def fail(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def format_conversation(messages):
    """Turn a list of {role, content} dicts into User:/Assistant: lines."""
    lines = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        # skip system/tool roles
    # Only keep conversations that actually alternate into a usable pair.
    if len(lines) < 2 or not lines[0].startswith("User: "):
        return ""
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/input.txt")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--max-conversations", type=int, default=20000,
                        help="Stop after this many conversations (0 = all). Keeps the corpus size manageable.")
    parser.add_argument("--min-chars", type=int, default=0,
                        help="Skip conversations shorter than this many characters.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        fail("the 'datasets' package is not installed. Run: python3 -m pip install datasets")

    try:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
    except Exception as e:
        fail(f"could not load {args.dataset} (split={args.split}): {e}. "
             "Check your network connection and that the dataset name/split are correct. "
             "If it is gated, run: huggingface-cli login")

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = args.out + ".tmp"

    written = 0
    chars = 0
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in ds:
                text = format_conversation(row.get("messages") or [])
                if not text or len(text) < args.min_chars:
                    continue
                f.write(text + "\n")
                written += 1
                chars += len(text)
                if written % 1000 == 0:
                    print(f"  {written} conversations...", flush=True)
                if args.max_conversations and written >= args.max_conversations:
                    break
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        fail(f"error while streaming the dataset: {e}")

    if written == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        fail("no usable conversations were produced. Check the dataset schema (expected a 'messages' list of role/content dicts).")

    os.replace(tmp_path, args.out)
    print(f"Wrote {written} conversations ({chars:,} characters) to {args.out}")


if __name__ == "__main__":
    main()
