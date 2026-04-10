"""
Training-throughput benchmark: Triton-fused kernels vs PyTorch baseline.

Measures
--------
- Tokens / second (end-to-end forward + backward + optimizer step)
- Peak GPU memory (MB)
- Steps / second
- Time to converge to a target loss (optional)

What gets compared
------------------
A. Baseline       — stock Qwen2.5 forward + standard CE
B. Triton full    — RMSNorm + SwiGLU + fused CE all patched
C. CE only        — only cross-entropy fused (isolates the biggest kernel)

Run
---
    python benchmarks/bench_training.py \
        --model_id Qwen/Qwen2.5-7B \
        --batch_size 4 \
        --seq_len 2048 \
        --steps 50 \
        --warmup_steps 5

Quick CPU-only smoke test (tiny model, no GPU needed):
    python benchmarks/bench_training.py \
        --model_id Qwen/Qwen2.5-0.5B \
        --batch_size 1 \
        --seq_len 128 \
        --steps 10 \
        --device cpu
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    tokens_per_sec: float
    steps_per_sec: float
    peak_memory_mb: float
    avg_loss: float
    std_loss: float

    def __str__(self) -> str:
        return (
            f"{self.name:<22} │ "
            f"{self.tokens_per_sec:>10,.1f} tok/s │ "
            f"{self.steps_per_sec:>8.2f} step/s │ "
            f"{self.peak_memory_mb:>8,.1f} MB │ "
            f"loss {self.avg_loss:.4f} ± {self.std_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# Fake dataset (synthetic random tokens)
# ---------------------------------------------------------------------------

class SyntheticDataset(torch.utils.data.Dataset):
    """Random token dataset for throughput benchmarks."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": ids, "labels": ids}


def collate(examples: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([e["input_ids"] for e in examples])
    labels = torch.stack([e["labels"] for e in examples])
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0


def _run_benchmark(
    name: str,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    warmup_steps: int,
    use_fused_ce: bool,
    ignore_index: int = -100,
) -> BenchResult:
    from kernels.cross_entropy import fused_cross_entropy

    model.train()
    losses: list[float] = []
    step_times: list[float] = []
    total_tokens = 0

    loader_iter = iter(loader)
    _reset_peak_memory(device)

    for step in range(steps + warmup_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits   # [B, T, V]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if use_fused_ce:
                loss = fused_cross_entropy(
                    shift_logits, shift_labels,
                    reduction="mean", ignore_index=ignore_index,
                )
            else:
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=ignore_index,
                )

        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_time = time.perf_counter() - t0

        if step >= warmup_steps:
            n_tokens = (shift_labels != ignore_index).sum().item()
            total_tokens += n_tokens
            step_times.append(step_time)
            losses.append(loss.item())

    total_elapsed = sum(step_times)
    result = BenchResult(
        name=name,
        tokens_per_sec=total_tokens / total_elapsed,
        steps_per_sec=len(step_times) / total_elapsed,
        peak_memory_mb=_peak_memory_mb(device),
        avg_loss=statistics.mean(losses),
        std_loss=statistics.stdev(losses) if len(losses) > 1 else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

def _load_model_and_apply_lora(
    model_id: str,
    load_in_4bit: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Load the model and apply LoRA, ready for optimisation."""
    from model.loader import load_qwen2
    from train.lora import LoRAConfig, apply_lora

    model, _ = load_qwen2(
        model_id,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map=str(device) if device.type == "cuda" else "cpu",
        gradient_checkpointing=False,  # off for benchmark repeatability
        attn_implementation="eager",   # consistent across configs
    )
    lora_cfg = LoRAConfig(r=8, alpha=16)
    model = apply_lora(model, lora_cfg)
    return model


def run_all_benchmarks(args: argparse.Namespace) -> list[BenchResult]:
    """Run all three benchmark configurations and return results."""
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    # We'll load the model once and apply/unapply patches per config.
    # Simpler: load three times (avoids state contamination).
    results: list[BenchResult] = []

    configs = [
        ("A  Baseline (no Triton)", False, False),
        ("B  Triton CE only",       False, True),
        ("C  Triton full",          True,  True),
    ]

    for name, patch_model_kernels, use_fused_ce in configs:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Running: {name}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model = _load_model_and_apply_lora(
            args.model_id, args.load_in_4bit, dtype, device
        )

        if patch_model_kernels:
            from model.patch import patch_model
            model = patch_model(model, verbose=False)

        from train.lora import get_lora_params
        optimizer = torch.optim.AdamW(
            get_lora_params(model),
            lr=2e-4,
            fused=device.type == "cuda",
        )

        vocab_size = model.config.vocab_size
        dataset = SyntheticDataset(vocab_size, args.seq_len, num_samples=500)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate,
            num_workers=0,
        )

        result = _run_benchmark(
            name=name,
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            use_fused_ce=use_fused_ce,
        )
        results.append(result)
        logger.info(f"Result: {result}")

        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[BenchResult], baseline: Optional[BenchResult] = None) -> None:
    header = (
        f"{'Config':<22} │ {'Tokens/sec':>12} │ "
        f"{'Steps/sec':>10} │ {'Peak Mem':>10} │ {'Loss':>16}"
    )
    print(f"\n{'Benchmark Results':═^{len(header)}}")
    print(header)
    print("─" * len(header))

    baseline = baseline or results[0]
    for r in results:
        speedup = r.tokens_per_sec / baseline.tokens_per_sec
        mem_delta = r.peak_memory_mb - baseline.peak_memory_mb
        print(
            f"{r.name:<22} │ "
            f"{r.tokens_per_sec:>10,.1f}   │ "
            f"{r.steps_per_sec:>8.2f}   │ "
            f"{r.peak_memory_mb:>8,.1f}   │ "
            f"loss {r.avg_loss:.4f} ± {r.std_loss:.4f}"
        )

    print()
    print("Speedup vs baseline:")
    for r in results[1:]:
        speedup = r.tokens_per_sec / baseline.tokens_per_sec
        mem_saved = baseline.peak_memory_mb - r.peak_memory_mb
        print(
            f"  {r.name}: {speedup:.2f}× faster, "
            f"{mem_saved:+.0f} MB memory (+ = saved)"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Triton-fused training kernels vs PyTorch baseline"
    )
    p.add_argument(
        "--model_id", default="Qwen/Qwen2.5-7B",
        help="HuggingFace model ID (use Qwen/Qwen2.5-0.5B for quick tests)"
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=50,
                   help="Measurement steps (after warmup)")
    p.add_argument("--warmup_steps", type=int, default=5,
                   help="Warmup steps excluded from timing")
    p.add_argument("--device", default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load base model in 4-bit NF4")
    p.add_argument("--output_json", default=None,
                   help="Path to write results as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(
        f"\nBenchmark config:\n"
        f"  model:       {args.model_id}\n"
        f"  batch_size:  {args.batch_size}\n"
        f"  seq_len:     {args.seq_len}\n"
        f"  device:      {args.device}\n"
        f"  steps:       {args.steps} (+{args.warmup_steps} warmup)\n"
    )

    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    results = run_all_benchmarks(args)
    print_report(results)

    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(
                [
                    {
                        "name": r.name,
                        "tokens_per_sec": r.tokens_per_sec,
                        "steps_per_sec": r.steps_per_sec,
                        "peak_memory_mb": r.peak_memory_mb,
                        "avg_loss": r.avg_loss,
                    }
                    for r in results
                ],
                f, indent=2,
            )
        logger.info(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()