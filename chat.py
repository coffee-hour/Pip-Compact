#!/usr/bin/env python3
"""Interactive CLI chat for Pip-Compact.

Loads out/ckpt.pt and out/tokenizer.json, then runs a terminal loop. Each turn is
formatted as "User: <msg>\nAssistant:" and the reply streams token by token until
the model writes a "\nUser:" stop sequence or --max_tokens is reached.

Usage:
    python chat.py
    python chat.py --ckpt out/ckpt.pt --tokenizer out/tokenizer.json --max_tokens 200

Commands inside the loop: /reset clears history, /quit exits.
"""
import argparse

import torch
import torch.nn.functional as F

from train import Pip, Config, BPE

STOP = "\nUser:"


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(ckpt_path="out/ckpt.pt", tok_path="out/tokenizer.json", device=None):
    """Return (model, tokenizer, device) ready for inference."""
    device = device or pick_device()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    model = Pip(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    model.to(device)
    tok = BPE.load(tok_path)
    return model, tok, device


@torch.no_grad()
def stream_reply(model, tok, device, prompt, max_tokens=200, temp=0.8, top_k=40):
    """Yield text chunks of the reply as they are generated.

    Stops when the decoded reply contains STOP or after max_tokens tokens. A short
    tail is held back so a partial STOP sequence is never printed.
    """
    ids = tok.encode(prompt) or [0]
    gen = []
    emitted = 0
    started = False
    for _ in range(max_tokens):
        window = torch.tensor([ids[-model.cfg.ctx:]], dtype=torch.long, device=device)
        logits, _, _ = model(window)
        logits = logits[:, -1].float() / temp
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        ids.append(nxt)
        gen.append(nxt)

        full = tok.decode(gen)
        if full.endswith("\ufffd"):  # incomplete UTF-8 sequence, wait for more bytes
            continue
        stop_at = full.find(STOP)
        if stop_at != -1:
            chunk = full[emitted:stop_at]
            if not started:
                chunk = chunk.lstrip()
            if chunk:
                yield chunk
            return
        safe = len(full) - (len(STOP) - 1)
        if safe > emitted:
            chunk = full[emitted:safe]
            if not started:
                chunk = chunk.lstrip()
            if chunk.strip():
                started = True
            emitted = safe
            if chunk:
                yield chunk

    # flush whatever is left (end of budget or trailing invalid bytes)
    full = tok.decode(gen)
    cut = full.find(STOP)
    if cut != -1:
        full = full[:cut]
    chunk = full[emitted:]
    if not started:
        chunk = chunk.lstrip()
    if chunk:
        yield chunk


def main():
    ap = argparse.ArgumentParser(description="Chat with Pip-Compact in the terminal")
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--tokenizer", default="out/tokenizer.json")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    a = ap.parse_args()

    model, tok, device = load(a.ckpt, a.tokenizer)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Pip-Compact ({n_params:.1f}M params) on {device}. /reset clears history, /quit exits.")

    history = ""
    while True:
        try:
            user = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            history = ""
            print("(history cleared)")
            continue

        prompt = history + f"User: {user}\nAssistant:"
        print("Assistant: ", end="", flush=True)
        parts = []
        for chunk in stream_reply(model, tok, device, prompt,
                                  max_tokens=a.max_tokens, temp=a.temp, top_k=a.top_k):
            parts.append(chunk)
            print(chunk, end="", flush=True)
        print()
        reply = "".join(parts).rstrip()
        history = prompt + " " + reply + "\n"


if __name__ == "__main__":
    main()
