# qwen2-triton-training

Fine-tune Qwen2.5-7B with LoRA + hand-written Triton kernels for the three
training bottlenecks: fused cross-entropy, RMSNorm, and SwiGLU.

---

## Results

**NVIDIA A100 80GB · Qwen2.5-7B · bf16 · seq=2048 · no gradient checkpointing**

| Config | Tokens/sec | Peak VRAM | vs baseline |
|---|---|---|---|
| A  Baseline (PyTorch) | 3,667 | 48,250 MB | — |
| B  Triton CE only | 3,622 | 47,654 MB | −596 MB |
| C  Triton full | 3,514 | **44,798 MB** | **−3,452 MB** |

The headline number is memory, not speed. At batch=1 the attention layers
dominate runtime and mask the kernel gains. The payoff is that **3.4 GB freed
at batch=1 becomes ~13 GB freed at batch=4** — enough to run a batch size
the baseline can't fit at all (it OOMs at batch=4). Larger batch is how
tokens/sec actually scales.

---

## Why it exists

Qwen2.5-7B has a vocabulary of 151,936 tokens. Standard PyTorch cross-entropy
materialises a `[M, 151936]` softmax tensor and keeps it alive for the backward
pass. At batch=4, seq=2048 that is ~5 GB of one intermediate alone.

The fused Triton CE kernel never writes the full softmax to HBM — it computes
loss and gradient in three passes over the logit row. Same math, no saved
activation.

---

## Kernels

| File | What it does |
|---|---|
| `kernels/cross_entropy.py` | Fused forward+backward CE. 3-pass: find max → log-sum-exp → write ∂L/∂logit. No softmax materialised. |
| `kernels/rms_norm.py` | RMSNorm fwd + bwd. Saves only `rstd` scalar per row, not the full normalised tensor. |
| `kernels/swiglu.py` | SiLU(gate)·up fwd + bwd. Gate, up, dout stay in SRAM registers. |

---

## Quickstart

```bash
uv sync                   # core deps
uv sync --extra dev       # + pytest

# correctness tests (no model download needed)
uv run pytest tests/test_kernels.py -v

# smoke test
uv run python benchmarks/bench_training.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --batch_size 2 --seq_len 128 --steps 10

# real benchmark
uv run python benchmarks/bench_training.py \
    --model_id Qwen/Qwen2.5-7B \
    --batch_size 1 --seq_len 2048 --steps 50
```

---

## Fine-tuning

```python
from model.loader  import load_qwen2
from model.patch   import patch_model
from train.lora    import LoRAConfig, apply_lora
from train.trainer import Trainer, TrainingConfig

model, tokenizer = load_qwen2("Qwen/Qwen2.5-7B", load_in_4bit=True,
                               gradient_checkpointing=True)
model = patch_model(model)
model = apply_lora(model, LoRAConfig(r=16, alpha=32))

Trainer(model, tokenizer, TrainingConfig(
    output_dir="./checkpoints",
    batch_size=4, grad_accum_steps=8,
    learning_rate=2e-4,
    use_fused_cross_entropy=True,
), train_dataset=dataset).train()
```

---

## Project layout

```
kernels/   cross_entropy.py · rms_norm.py · swiglu.py
model/     loader.py · patch.py
train/     lora.py · trainer.py
benchmarks/bench_training.py
tests/     test_kernels.py
```

40/40 kernel correctness tests pass against PyTorch reference implementations.