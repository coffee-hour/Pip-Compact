# Pip-Compact

Pip-Compact is a ~90M-parameter Pip-style transformer for one 16GB T4. It is one self-contained training file (train.py) with a pure-Python byte-level BPE tokenizer, a Pre-LN transformer (SwiGLU or GELU MLP, tied embeddings), satellite subnets (endocrine, thalamus, hippocampus, router, critic), fp16 autocast, gradient accumulation, gradient checkpointing, and checkpoint/resume.

## One-command training

```
bash train.sh
```

The launcher does the following:
- installs torch if it is missing,
- downloads TinyShakespeare to data/input.txt if the corpus is missing,
- picks the device: CUDA, then MPS, then CPU,
- uses fp16 autocast on CUDA and fp32 on MPS and CPU (fp16 needs CUDA),
- resumes from out/ckpt.pt when that file exists.

Extra flags are passed to train.py, for example `bash train.sh --steps 8000`.

## Files

- train.py: model, tokenizer, training loop, checkpointing, and sampling.
- train.sh: the launcher above.
- index.html: browser visualizer with subnets, telemetry, and a console preview.

## Status

Code only. No pretrained weights are included.
