# Pip-Compact

An optimized, lightweight companion to Big Pip. Pip-Compact is a Pre-LayerNorm decoder-only transformer with satellite subnets, sized to train smoothly on a single 16 GB NVIDIA T4 in Kaggle or Google Colab without running out of memory.

## Kaggle or Colab: one-click start

1. Create a Kaggle notebook (or Colab) and set the accelerator to GPU T4.
2. Run these cells:

```bash
!git clone https://github.com/coffee-hour/Pip-Compact.git
%cd Pip-Compact
!bash train.sh --steps 5000
```

## Local, Mac, or CPU

```bash
bash train.sh --steps 5000
```

Notes:
- `train.sh` installs torch if it is missing, then picks the device: CUDA, then MPS, then CPU.
- Mixed precision: fp16 autocast on CUDA (the T4 default). MPS and CPU run in fp32. fp16 is never passed on a non-CUDA device.
- With no corpus present, `train.sh` downloads TinyShakespeare to `data/input.txt`. To train on your own data, put a UTF-8 text file at `data/input.txt` or set `DATA=/path/to/file.txt`.
- Checkpoints go to `out/ckpt.pt`, and the trained BPE tokenizer goes to `out/tokenizer.json`. `train.sh` resumes from `out/ckpt.pt` automatically when that file exists.

## Files

- `train.py`: self-contained trainer. It includes the pure-Python byte-level BPE tokenizer, the model, the satellite subnets, mixed precision, gradient accumulation, checkpointing, and sampling. It depends only on PyTorch.
- `train.sh`: one-command launcher. It installs torch if needed, fetches a default corpus if none is present, selects the device, and resumes automatically.
- `index.html`: zero-build browser architecture visualizer. Open it directly in a browser. It shows the parameter budget and a block diagram that updates as you change the config.

## Useful flags

- `--fp16` or `--bf16`: mixed precision. `--fp16` is the T4 choice. On CPU or MPS the launcher runs fp32 and tells you why.
- `--batch` and `--accum`: micro-batch size and gradient accumulation steps. The effective batch is their product. Defaults are 8 and 4.
- `--grad_ckpt`: activation checkpointing. Less VRAM, slower steps. Turn it on if you hit OOM.
- `--ctx`: context length, default 512.
- `--act swiglu|gelu`: MLP variant, default swiglu.
- `--resume`: continue from `out/ckpt.pt`. Added automatically by `train.sh` when the checkpoint exists.

## Architecture

| Item | Value |
|---|---|
| Layers | 12 |
| Attention heads | 12 (head dim 64) |
| Width (n_embd) | 768 |
| Context | 512 |
| Vocabulary | 2048 (byte-level BPE; 256 byte tokens plus 1792 learned merges) |
| Normalization | Pre-LayerNorm |
| MLP | SwiGLU, hidden 2048 (or GELU, hidden 3072) |
| Embeddings | Tied input embedding and LM head |
| Total parameters | about 90.2M with the defaults |

Block layout, in order:
1. LayerNorm, then causal multi-head attention, scaled by a per-layer endocrine gain.
2. Thalamus relay: a low-rank residual (768 to 48 to 768) that gates information flow.
3. LayerNorm, then a router-gated MLP. A per-token sigmoid router scales the MLP output.

Satellite subnets:
- Endocrine: a linear map from the mean token state to per-layer, per-example gains (2 times sigmoid, mean 1 at init).
- Thalamus: the low-rank relay inside each block.
- Hippocampus: 16 learned memory slots that every position can attend to. Its output gate starts at zero, so it begins as an identity.
- Router: a per-token gate on each MLP output.
- Critic: an auxiliary head that predicts per-token cross-entropy. It adds 0.01 times MSE to the loss, and its input is detached from the LM loss path.

Note on size: with this configuration (vocab 2048, width 768, 12 layers) the model has about 90M parameters, not 125M. A 125M model at this width and depth would need a larger vocabulary or more layers. The index.html visualizer shows the exact budget for any setting.

## Memory

At the defaults (micro-batch 8, context 512, fp16 autocast, AdamW), training is intended to fit on a 16 GB T4. This has not been measured on hardware yet. If you see OOM, lower `--batch`, raise `--accum` to keep the effective batch, or add `--grad_ckpt`.
