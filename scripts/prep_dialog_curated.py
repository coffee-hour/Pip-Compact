#!/usr/bin/env python3
"""Curated dialog prep for Pip-Compact. Writes data/input.txt in the same User:/Assistant: format as prep_dialog.py.

Filters UltraChat (HuggingFaceH4/ultrachat_200k) for short, clean, well-formed exchanges that a ~90M model can learn
from: capped turns and length, no boilerplate or markup, no non-ASCII junk, no near-duplicate openers, and every
conversation must alternate User/Assistant. Optionally prepends the hand-written seed file data/curated_seed.txt.

Usage: python scripts/prep_dialog_curated.py [--out data/input.txt] [--max-conversations 8000]
       [--seed data/curated_seed.txt] [--max-turns 3] [--max-reply-chars 900]
"""
import argparse
import os
import re
import sys

BAD_PATTERNS = re.compile(
    r"(as an ai language model|i am unable to browse|i cannot browse|knowledge cutoff|"
    r"<\/?[a-z][^>]*>|```|\bhttps?://|\bwww\.|\[\d+\]|\*\*|^#{1,6}\s)",
    re.IGNORECASE | re.MULTILINE,
)
WS = re.compile(r"[ \t]+")


def clean(text):
    """Normalize whitespace and keep printable ASCII only (matches train.py's byte-level BPE expectations)."""
    text = text.replace("\r", "\n")
    text = "".join(ch if (ch == "\n" or 32 <= ord(ch) < 127) else " " for ch in text)
    text = WS.sub(" ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


def format_pairs(messages, max_turns, max_reply_chars):
    """Return a formatted conversation string, or '' if it does not meet the quality bar."""
    turns = []
    for m in messages:
        role = (m.get("role") or "").lower()
        if role not in ("user", "assistant"):
            return ""  # drop system/tool turns entirely
        content = clean(m.get("content") or "")
        if not content:
            return ""
        turns.append((role, content))
    # must strictly alternate, starting with user and ending with assistant
    if not turns or turns[0][0] != "user" or turns[-1][0] != "assistant":
        return ""
    for i, (role, _) in enumerate(turns):
        if role != ("user" if i % 2 == 0 else "assistant"):
            return ""
    if len(turns) // 2 > max_turns:
        return ""
    lines = []
    for role, content in turns:
        if role == "user":
            if len(content) > 600:
                return ""
            lines.append(f"User: {content}")
        else:
            if len(content) > max_reply_chars or BAD_PATTERNS.search(content):
                return ""
            lines.append(f"Assistant: {content}")
    if any(BAD_PATTERNS.search(l) for l in lines):
        return ""
    return "\n".join(lines) + "\n\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/input.txt")
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--split", default="train_sft")
    ap.add_argument("--seed", default="data/curated_seed.txt", help="hand-written dialogs prepended if present")
    ap.add_argument("--max-conversations", type=int, default=8000)
    ap.add_argument("--max-turns", type=int, default=3, help="max user/assistant pairs per conversation")
    ap.add_argument("--max-reply-chars", type=int, default=900)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: install datasets first: python3 -m pip install datasets")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    seen_openers = set()
    kept = dropped = 0
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            if args.seed and os.path.exists(args.seed):
                with open(args.seed, encoding="utf-8") as s:
                    seed = s.read().strip()
                f.write(seed + "\n\n")
                print(f"seed: {args.seed} ({seed.count(chr(10) + chr(10)) + 1} dialogs)")
            ds = load_dataset(args.dataset, split=args.split, streaming=True)
            for row in ds:
                text = format_pairs(row.get("messages") or [], args.max_turns, args.max_reply_chars)
                if not text:
                    dropped += 1
                    continue
                opener = text.split("\n", 1)[0][:40].lower()
                if opener in seen_openers:  # cheap near-duplicate guard on the first user message
                    dropped += 1
                    continue
                seen_openers.add(opener)
                f.write(text)
                kept += 1
                if kept % 1000 == 0:
                    print(f"  {kept} kept, {dropped} dropped", flush=True)
                if kept >= args.max_conversations:
                    break
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        sys.exit(f"ERROR: {e}")
    if kept == 0:
        os.remove(tmp)
        sys.exit("ERROR: no conversations passed the filters")
    os.replace(tmp, args.out)
    print(f"Wrote {kept} conversations to {args.out} ({dropped} dropped by filters)")


if __name__ == "__main__":
    main()
