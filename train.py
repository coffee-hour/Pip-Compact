#!/usr/bin/env python3
"""Pip-Compact: a ~90M-parameter Pip-style transformer for one 16GB T4.

Single self-contained file: pure-Python byte-level BPE tokenizer, Pre-LN transformer
(SwiGLU or GELU MLP, tied embeddings), satellite subnets (endocrine, thalamus,
hippocampus, router, critic), fp16/bf16 autocast, gradient accumulation,
gradient checkpointing, checkpoint/resume.
"""
import argparse, contextlib, json, math, os, re, time
from collections import Counter
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PRETOK = re.compile(r"\s?\w+|\s?[^\w\s]+|\s+")


# ---------------------------------------------------------------- tokenizer
class BPE:
    """Byte-level BPE. Ids 0-255 are raw bytes; id 256+r is the r-th merge."""

    def __init__(self, merges=()):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]

    @property
    def n_vocab(self):
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size):
        words = Counter(tuple(w.encode("utf-8")) for w in PRETOK.findall(text))
        merges = []
        while 256 + len(merges) < vocab_size:
            pairs = Counter()
            for w, c in words.items():
                for p in zip(w, w[1:]):
                    pairs[p] += c
            if not pairs:
                break
            best, cnt = pairs.most_common(1)[0]
            if cnt < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            nw = Counter()
            for w, c in words.items():
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                nw[tuple(out)] += c
            words = nw
        return cls(merges)

    def encode(self, text):
        ids = []
        for w in PRETOK.findall(text):
            toks = list(w.encode("utf-8"))
            while len(toks) > 1:
                best_r, best_i = None, None
                for i in range(len(toks) - 1):
                    r = self.ranks.get((toks[i], toks[i + 1]))
                    if r is not None and (best_r is None or r < best_r):
                        best_r, best_i = r, i
                if best_r is None:
                    break
                toks[best_i:best_i + 2] = [256 + best_r]
            ids.extend(toks)
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["merges"])


# ---------------------------------------------------------------- model
@dataclass
class Config:
    vocab: int = 2048
    ctx: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    act: str = "swiglu"      # "swiglu" or "gelu"
    mem_slots: int = 16      # hippocampus memory slots
    critic_w: float = 0.01   # weight of the critic auxiliary loss


class Attn(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.h = c.n_head
        s.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        s.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, s.h, C // s.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(s, c):
        super().__init__()
        C = c.n_embd
        s.swiglu = c.act == "swiglu"
        H = (int(8 * C / 3) // 64) * 64 if s.swiglu else 4 * C
        s.w1 = nn.Linear(C, H, bias=False)
        s.w2 = nn.Linear(H, C, bias=False)
        s.w3 = nn.Linear(C, H, bias=False) if s.swiglu else None

    def forward(s, x):
        if s.swiglu:
            return s.w2(F.silu(s.w1(x)) * s.w3(x))
        return s.w2(F.gelu(s.w1(x)))


class Relay(nn.Module):
    """Thalamus: low-rank relay that gates information flow between blocks."""

    def __init__(s, C):
        super().__init__()
        s.ln = nn.LayerNorm(C)
        s.down = nn.Linear(C, C // 16, bias=False)
        s.up = nn.Linear(C // 16, C, bias=False)

    def forward(s, x):
        return s.up(F.gelu(s.down(s.ln(x))))


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        C = c.n_embd
        s.ln1 = nn.LayerNorm(C)
        s.attn = Attn(c)
        s.relay = Relay(C)                 # thalamus
        s.ln2 = nn.LayerNorm(C)
        s.mlp = MLP(c)
        s.route = nn.Linear(C, 1)          # router: per-token mixing of MLP path

    def forward(s, x, g):
        # g: per-example endocrine gain for this layer, shape (B,)
        x = x + g.view(-1, 1, 1) * s.attn(s.ln1(x))
        x = x + s.relay(x)
        x = x + torch.sigmoid(s.route(x)) * s.mlp(s.ln2(x))
        return x


class Hippo(nn.Module):
    """Hippocampus: learned memory slots that every position can read (non-causal, no leakage)."""

    def __init__(s, C, h, M):
        super().__init__()
        s.h = h
        s.mem = nn.Parameter(torch.randn(M, C) * 0.02)
        s.ln = nn.LayerNorm(C)
        s.q = nn.Linear(C, C, bias=False)
        s.kv = nn.Linear(C, 2 * C, bias=False)
        s.o = nn.Linear(C, C, bias=False)
        s.gate = nn.Parameter(torch.zeros(1))  # starts as identity

    def forward(s, x):
        B, T, C = x.shape
        h, d = s.h, C // s.h
        q = s.q(s.ln(x)).view(B, T, h, d).transpose(1, 2)
        m = s.mem.unsqueeze(0).expand(B, -1, -1)
        k, v = s.kv(m).view(B, -1, 2, h, d).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return x + torch.tanh(s.gate) * s.o(y.transpose(1, 2).reshape(B, T, C))


class Pip(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.cfg = c
        C = c.n_embd
        s.tok = nn.Embedding(c.vocab, C)
        s.pos = nn.Embedding(c.ctx, C)
        s.endo = nn.Linear(C, c.n_layer)            # endocrine: per-layer hormone gains
        s.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        s.hippo = Hippo(C, c.n_head, c.mem_slots)   # hippocampus
        s.ln_f = nn.LayerNorm(C)
        s.critic = nn.Linear(C, 1)                  # critic: predicts per-token loss
        s.head = nn.Linear(C, c.vocab, bias=False)
        s.head.weight = s.tok.weight                # tied embeddings
        s.grad_ckpt = False
        s.apply(s._init)
        for n, p in s.named_parameters():           # GPT-2 style residual scaling
            if n.endswith("attn.proj.weight") or n.endswith("mlp.w2.weight"):
                with torch.no_grad():
                    p.mul_((2 * c.n_layer) ** -0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(s, idx, targets=None):
        B, T = idx.shape
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        g = 2 * torch.sigmoid(s.endo(x.mean(1)))  # (B, n_layer), mean 1 at init
        for i, blk in enumerate(s.blocks):
            if s.grad_ckpt and s.training:
                x = checkpoint(blk, x, g[:, i], use_reentrant=False)
            else:
                x = blk(x, g[:, i])
        x = s.ln_f(s.hippo(x))
        logits = s.head(x)
        if targets is None:
            return logits, None, None
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                             targets.view(-1), reduction="none")
        value = s.critic(x).float().view(-1)
        lm = ce.mean()
        loss = lm + s.cfg.critic_w * F.mse_loss(value, ce.detach())
        return logits, loss, lm


# ---------------------------------------------------------------- training utils
def get_batch(data, B, T, device):
    ix = torch.randint(len(data) - T - 1, (B,)).tolist()
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + T] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, a, cfg, device, amp_fn):
    model.eval()
    tot = 0.0
    for _ in range(a.eval_iters):
        x, y = get_batch(data, a.batch, cfg.ctx, device)
        with amp_fn():
            _, _, lm = model(x, y)
        tot += lm.item()
    model.train()
    return tot / a.eval_iters


@torch.no_grad()
def generate(model, idx, n, device, amp_fn, temp=0.8, top_k=40):
    ids = torch.tensor([idx], dtype=torch.long, device=device)
    model.eval()
    for _ in range(n):
        with amp_fn():
            logits, _, _ = model(ids[:, -model.cfg.ctx:])
        logits = logits[:, -1].float() / temp
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[0].tolist()


def save_ckpt(path, model, opt, scaler, step, cfg):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "cfg": asdict(cfg)},
               path + ".tmp")
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train Pip-Compact on a single GPU")
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8, help="micro-batch size")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--vocab", type=int, default=2048, help="target BPE vocab size")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--act", choices=["swiglu", "gelu"], default="swiglu")
    ap.add_argument("--tok_chars", type=int, default=5_000_000, help="chars used to train BPE")
    ap.add_argument("--fp16", action="store_true", help="float16 AMP (use on T4)")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 AMP (Ampere or newer)")
    ap.add_argument("--grad_ckpt", action="store_true", help="activation checkpointing (less VRAM, slower)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sample", default="The ")
    ap.add_argument("--gen_tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if a.fp16 and a.bf16:
        ap.error("choose --fp16 or --bf16, not both")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (a.fp16 or a.bf16) and device != "cuda":
        ap.error("--fp16/--bf16 need a CUDA GPU")
    if a.bf16 and not torch.cuda.is_bf16_supported():
        ap.error("--bf16 needs an Ampere-or-newer GPU; the T4 should use --fp16")
    use_amp = a.fp16 or a.bf16
    amp_dtype = torch.bfloat16 if a.bf16 else torch.float16

    def amp_fn():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return contextlib.nullcontext()

    scaler = torch.amp.GradScaler("cuda", enabled=a.fp16)
    os.makedirs(a.out, exist_ok=True)

    # --- data + tokenizer
    with open(a.data, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tok_path = os.path.join(a.out, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = BPE.load(tok_path)
    else:
        print(f"training BPE on {min(len(text), a.tok_chars)} chars, target vocab {a.vocab}")
        tok = BPE.train(text[:a.tok_chars], a.vocab)
        tok.save(tok_path)
    cache = os.path.join(a.out, "tokens.pt")
    if os.path.exists(cache):
        ids = torch.load(cache)
    else:
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        torch.save(ids, cache)
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]
    print(f"tokens: {len(ids):,} (train {len(train_ids):,}, val {len(val_ids):,}), vocab {tok.n_vocab}")

    # --- model
    cfg = Config(vocab=tok.n_vocab, ctx=a.ctx, n_layer=a.n_layer,
                 n_head=a.n_head, n_embd=a.n_embd, act=a.act)
    model = Pip(cfg).to(device)
    model.grad_ckpt = a.grad_ckpt
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f}M on {device}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95), fused=(device == "cuda"))

    ckpt_path = os.path.join(a.out, "ckpt.pt")
    start = 0
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["step"]
        print(f"resumed from step {start}")

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

    # --- loop
    model.train()
    t0 = time.time()
    for step in range(start, a.steps):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        for _ in range(a.accum):
            x, y = get_batch(train_ids, a.batch, cfg.ctx, device)
            with amp_fn():
                _, loss, lm = model(x, y)
            scaler.scale(loss / a.accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        s1 = step + 1
        if s1 % 20 == 0 or s1 == start + 1:
            print(f"step {s1}/{a.steps} lm_loss {lm.item():.4f} lr {lr_at(step):.2e} "
                  f"{time.time() - t0:.0f}s")
        if s1 % a.eval_every == 0 or s1 == a.steps:
            vl = evaluate(model, val_ids, a, cfg, device, amp_fn)
            mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
            print(f"  val_lm_loss {vl:.4f} ppl {math.exp(vl):.2f} peak_vram {mem:.2f} GiB")
        if s1 % a.save_every == 0 or s1 == a.steps:
            save_ckpt(ckpt_path, model, opt, scaler, s1, cfg)
            print(f"  checkpoint saved to {ckpt_path}")

    # --- sample
    out_ids = generate(model, tok.encode(a.sample) or [0], a.gen_tokens, device, amp_fn)
    print("--- sample ---")
    print(tok.decode(out_ids))


if __name__ == "__main__":
    main()
