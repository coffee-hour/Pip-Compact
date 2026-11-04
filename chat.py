#!/usr/bin/env python3
"""Interactive CLI chat for Pip-Compact.

Loads out/ckpt.pt and out/tokenizer.json, then runs a terminal loop. Each turn is
formatted as "User: <msg>\nAssistant:" and the reply streams token by token until
the model writes a "\nUser:" stop sequence or --max_tokens is reached.

Decoding is byte-safe: an incremental UTF-8 decoder holds back partial multi-byte
sequences until they complete, and any replacement characters are stripped before
they reach the screen or the chat history.

With --web, each user turn is preceded by live DuckDuckGo results (see web_search.py,
standard library only, no API keys). Search text is added to that turn's prompt only;
the chat history keeps the bare conversation.

Usage:
    python chat.py
    python chat.py --ckpt out/ckpt.pt --tokenizer out/tokenizer.json --max_tokens 200
    python chat.py --web --web_results 3

Commands inside the loop: /reset clears history, /quit exits.
"""
import argparse
import codecs
from dataclasses import fields

import torch

from train import Pip, Config, BPE

STOP = "\nUser:"
REPLACEMENT = "\ufffd"
REP_WINDOW = 256  # only the most recent reply tokens are penalized


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(ckpt_path="out/ckpt.pt", tok_path="out/tokenizer.json", device=None):
    """Return (model, tokenizer, device) ready for inference."""
    device = device or pick_device()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in known})
    model = Pip(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    model.to(device)
    tok = BPE.load(tok_path)
    if tok.n_vocab != cfg.vocab:
        raise ValueError(f"tokenizer vocab {tok.n_vocab} does not match checkpoint vocab {cfg.vocab}")
    return model, tok, device


def sample_token(logits, recent, temp, top_k, top_p, rep_penalty):
    """Pick one token id from a 1-D logits tensor.

    Applies a repetition penalty to tokens in `recent`, then temperature, top-k and top-p.
    temp <= 0 means greedy decoding.
    """
    logits = logits.float().clone()
    if rep_penalty != 1.0 and recent:
        idx = torch.tensor(sorted(set(recent)), dtype=torch.long, device=logits.device)
        vals = logits[idx]
        logits[idx] = torch.where(vals > 0, vals / rep_penalty, vals * rep_penalty)
    if temp <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temp
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < v[-1]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=0)
        sp[(cum - sp) > top_p] = 0.0  # keep the smallest set whose mass reaches top_p
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def stream_reply(model, tok, device, prompt, max_tokens=200, temp=0.8, top_k=40,
                 top_p=0.9, rep_penalty=1.15):
    """Yield clean text chunks of the reply as they are generated.

    Stops when the decoded reply contains STOP or after max_tokens tokens. A short
    tail is held back so a partial STOP sequence is never printed.
    """
    ids = tok.encode(prompt) or [0]
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    reply_ids = []
    text = ""          # decoded reply so far (only complete characters)
    emitted = 0        # how much of `text` has been yielded
    started = False    # whether any non-whitespace text has been yielded

    def take(chunk):
        nonlocal started
        chunk = chunk.replace(REPLACEMENT, "")
        if not started:
            chunk = chunk.lstrip()
        if chunk.strip():
            started = True
        return chunk

    for _ in range(max_tokens):
        window = torch.tensor([ids[-model.cfg.ctx:]], dtype=torch.long, device=device)
        logits, _, _ = model(window)
        nxt = sample_token(logits[0, -1], reply_ids[-REP_WINDOW:], temp, top_k, top_p, rep_penalty)
        ids.append(nxt)
        reply_ids.append(nxt)
        text += decoder.decode(tok.vocab[nxt])  # holds partial UTF-8 until complete

        stop_at = text.find(STOP)
        if stop_at != -1:
            chunk = take(text[emitted:stop_at])
            if chunk:
                yield chunk
            return
        safe = len(text) - (len(STOP) - 1)
        if safe > emitted:
            chunk = take(text[emitted:safe])
            emitted = safe
            if chunk:
                yield chunk

    # end of budget: flush any incomplete trailing bytes and the held-back tail
    text += decoder.decode(b"", final=True)
    stop_at = text.find(STOP)
    if stop_at != -1:
        text = text[:stop_at]
    chunk = take(text[emitted:])
    if chunk:
        yield chunk


def build_prompt(history, user, web_context=""):
    """Return (prompt_for_model, bare_turn). Web context, if any, goes only into the prompt."""
    bare_turn = history + f"User: {user}\nAssistant:"
    if web_context:
        prompt = history + f"Web search results:\n{web_context}\n\nUser: {user}\nAssistant:"
    else:
        prompt = bare_turn
    return prompt, bare_turn


def main():
    ap = argparse.ArgumentParser(description="Chat with Pip-Compact in the terminal")
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--tokenizer", default="out/tokenizer.json")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--rep_penalty", type=float, default=1.15, help="1.0 disables")
    ap.add_argument("--web", action="store_true", help="inject live DuckDuckGo results into each prompt")
    ap.add_argument("--web_results", type=int, default=3, help="number of search results to inject")
    a = ap.parse_args()

    fetch_context = None
    if a.web:
        from web_search import fetch_context  # stdlib only; imported only when requested

    model, tok, device = load(a.ckpt, a.tokenizer)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    web_note = " Web search on." if a.web else ""
    print(f"Pip-Compact ({n_params:.1f}M params) on {device}.{web_note} /reset clears history, /quit exits.")

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

        web_context = fetch_context(user, max_results=a.web_results) if a.web else ""
        prompt, bare_turn = build_prompt(history, user, web_context)
        print("Assistant: ", end="", flush=True)
        parts = []
        for chunk in stream_reply(model, tok, device, prompt, max_tokens=a.max_tokens,
                                  temp=a.temp, top_k=a.top_k, top_p=a.top_p,
                                  rep_penalty=a.rep_penalty):
            parts.append(chunk)
            print(chunk, end="", flush=True)
        print()
        reply = "".join(parts).replace(REPLACEMENT, "").rstrip()
        history = bare_turn + " " + reply + "\n"


if __name__ == "__main__":
    main()
