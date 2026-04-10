"""
Training loop with gradient accumulation and Triton-fused cross-entropy.

Features
--------
- Gradient accumulation over N micro-batches before each optimizer step
- bfloat16 / float16 mixed-precision (no GradScaler needed for bf16)
- Fused cross-entropy loss (avoids materialising the [M, V] softmax)
- LoRA-aware: only LoRA parameters receive gradient updates
- Cosine LR schedule with linear warmup
- Periodic checkpointing (LoRA weights only — small files)
- WandB / TensorBoard logging (optional)
- Gradient clipping

USAGE
-----
    from model.loader import load_qwen2
    from model.patch  import patch_model
    from train.lora   import LoRAConfig, apply_lora
    from train.trainer import Trainer, TrainingConfig

    model, tokenizer = load_qwen2("Qwen/Qwen2.5-7B")
    model = patch_model(model)
    model = apply_lora(model, LoRAConfig(r=16, alpha=32))

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=TrainingConfig(
            output_dir="./checkpoints",
            num_epochs=3,
            batch_size=4,
            grad_accum_steps=8,
            learning_rate=2e-4,
        ),
        train_dataset=my_dataset,
    )
    trainer.train()
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kernels.cross_entropy import fused_cross_entropy
from train.lora import get_lora_params, lora_state_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All training hyperparameters in one place."""

    output_dir: str = "./checkpoints"

    # ── Optimisation ─────────────────────────────────────────────────────
    num_epochs: int = 3
    batch_size: int = 4
    """Micro-batch size (per GPU, per accumulation step)."""
    grad_accum_steps: int = 8
    """Effective batch = batch_size × grad_accum_steps."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ── Schedule ─────────────────────────────────────────────────────────
    warmup_ratio: float = 0.03
    lr_schedule: str = "cosine"     # "cosine" | "linear" | "constant"

    # ── Sequence ─────────────────────────────────────────────────────────
    max_seq_len: int = 2048

    # ── Mixed precision ───────────────────────────────────────────────────
    dtype: torch.dtype = torch.bfloat16
    """Training dtype.  bf16 preferred (no GradScaler, better range than fp16)."""

    # ── Loss ─────────────────────────────────────────────────────────────
    use_fused_cross_entropy: bool = True
    """Use the Triton fused CE kernel. Disable to benchmark against baseline."""

    ignore_index: int = -100
    """Tokens with this label are excluded from the loss (e.g. padding, prompt)."""

    # ── Checkpointing ─────────────────────────────────────────────────────
    save_steps: int = 500
    save_total_limit: int = 3
    """Keep only the N most recent checkpoints."""

    # ── Logging ──────────────────────────────────────────────────────────
    log_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = "qwen2-triton-training"
    wandb_run_name: Optional[str] = None

    # ── DataLoader ────────────────────────────────────────────────────────
    num_workers: int = 4
    pin_memory: bool = True

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int = 42


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine schedule with linear warmup and a minimum LR floor."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        return max(
            0.0,
            float(num_training_steps - current_step)
            / max(1, num_training_steps - num_warmup_steps),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Minimal training loop designed for LoRA fine-tuning of Qwen2.5-7B
    with Triton-fused training ops.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or self._default_collator

        torch.manual_seed(config.seed)
        self.device = next(
            (p.device for p in model.parameters()), torch.device("cuda")
        )

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_wandb()

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_wandb(self) -> None:
        self._wandb = None
        if self.config.use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=vars(self.config),
                )
            except ImportError:
                logger.warning("wandb not installed; disabling W&B logging.")

    def _build_optimizer(self) -> torch.optim.AdamW:
        """AdamW on LoRA parameters only."""
        params = get_lora_params(self.model)
        if not params:
            logger.warning(
                "No trainable parameters found!  Did you call apply_lora()?"
            )
        return torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),   # fused AdamW on CUDA
        )

    def _build_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

    def _default_collator(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        """
        Basic collator: pad input_ids and labels to max length in batch.
        Expects each example to have 'input_ids' and optionally 'labels'.
        """
        max_len = min(
            max(len(e["input_ids"]) for e in examples),
            self.config.max_seq_len,
        )
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.full(
            (len(examples), max_len), pad_id, dtype=torch.long
        )
        labels = torch.full(
            (len(examples), max_len), self.config.ignore_index, dtype=torch.long
        )
        attention_mask = torch.zeros(len(examples), max_len, dtype=torch.long)

        for i, ex in enumerate(examples):
            ids = ex["input_ids"][: max_len]
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
            lab = ex.get("labels", ids)[: max_len]
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # ── Loss computation ─────────────────────────────────────────────────

    def _compute_loss(
        self,
        logits: torch.Tensor,   # [B, T, V]
        labels: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss, optionally using the fused Triton kernel.

        The fused kernel avoids materialising the full [B*T, V] softmax tensor,
        saving up to ~2 GB for vocab_size=150k at batch=4, seq=2048.
        """
        if self.config.use_fused_cross_entropy:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()    # [B, T-1, V]
            shift_labels = labels[..., 1:].contiguous()         # [B, T-1]
            return fused_cross_entropy(
                shift_logits, shift_labels,
                reduction="mean",
                ignore_index=self.config.ignore_index,
            )
        else:
            # Baseline: standard PyTorch CE (materialises full softmax)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            return torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=self.config.ignore_index,
            )

    # ── Training step ────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _maybe_no_sync(self, condition: bool):
        """Context manager for gradient accumulation with DDP."""
        if condition and hasattr(self.model, "no_sync"):
            with self.model.no_sync():
                yield
        else:
            yield

    def _train_step(
        self,
        batch: dict[str, torch.Tensor],
        accum_step: int,
        num_accum: int,
    ) -> torch.Tensor:
        """
        Forward + backward for one micro-batch.
        Scales loss by 1/num_accum for correct gradient accumulation.
        """
        is_last_accum = (accum_step == num_accum - 1)

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.amp.autocast("cuda", dtype=self.config.dtype):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,       # disable KV cache during training
            )
            logits = outputs.logits

            loss = self._compute_loss(logits, labels) / num_accum

        # Skip sync on non-final accumulation steps (DDP efficiency)
        with self._maybe_no_sync(not is_last_accum):
            loss.backward()

        return loss.detach() * num_accum   # unscaled for logging

    # ── Eval ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation and return metrics."""
        if self.eval_dataset is None:
            return {}

        self.model.eval()
        loader = self._build_dataloader(self.eval_dataset, shuffle=False)
        total_loss = 0.0
        total_tokens = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.config.dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                loss = self._compute_loss(outputs.logits, labels)

            n_tokens = (labels != self.config.ignore_index).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        self.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        return {"eval_loss": avg_loss, "eval_perplexity": math.exp(avg_loss)}

    # ── Checkpointing ────────────────────────────────────────────────────

    def _save_checkpoint(self, step: int) -> None:
        """Save LoRA adapter weights only (not the full model)."""
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(exist_ok=True)

        # LoRA weights
        adapter_weights = lora_state_dict(self.model)
        torch.save(adapter_weights, ckpt_dir / "adapter_model.pt")

        # Training state (for resuming)
        torch.save(
            {
                "step": step,
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler.state_dict(),
            },
            ckpt_dir / "training_state.pt",
        )

        logger.info(f"Saved checkpoint at step {step} → {ckpt_dir}")

        # Prune old checkpoints
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        ckpts = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        while len(ckpts) > self.config.save_total_limit:
            oldest = ckpts.pop(0)
            import shutil
            shutil.rmtree(oldest)
            logger.info(f"Removed old checkpoint: {oldest}")

    # ── Main training loop ───────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        """
        Run the full training loop.

        Returns:
            dict with training statistics.
        """
        cfg = self.config
        self.model.train()

        train_loader = self._build_dataloader(self.train_dataset)
        steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
        num_training_steps = steps_per_epoch * cfg.num_epochs
        num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)

        self._optimizer = self._build_optimizer()

        if cfg.lr_schedule == "cosine":
            self._scheduler = get_cosine_schedule_with_warmup(
                self._optimizer, num_warmup_steps, num_training_steps
            )
        elif cfg.lr_schedule == "linear":
            self._scheduler = get_linear_schedule_with_warmup(
                self._optimizer, num_warmup_steps, num_training_steps
            )
        else:
            self._scheduler = torch.optim.lr_scheduler.ConstantLR(
                self._optimizer, factor=1.0
            )

        logger.info(
            f"\n{'Training configuration':─^60}\n"
            f"  Epochs:              {cfg.num_epochs}\n"
            f"  Steps/epoch:         {steps_per_epoch}\n"
            f"  Total steps:         {num_training_steps}\n"
            f"  Warmup steps:        {num_warmup_steps}\n"
            f"  Effective batch:     {cfg.batch_size * cfg.grad_accum_steps}\n"
            f"  Fused CE:            {cfg.use_fused_cross_entropy}\n"
            f"{'─' * 60}"
        )

        global_step = 0
        total_tokens = 0
        running_loss = 0.0
        t_start = time.perf_counter()

        pbar = tqdm(total=num_training_steps, desc="Training")

        for epoch in range(cfg.num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                accum_step = batch_idx % cfg.grad_accum_steps

                # Forward + backward (scaled)
                loss = self._train_step(batch, accum_step, cfg.grad_accum_steps)
                running_loss += loss.item()

                # Count non-padding tokens
                total_tokens += (
                    batch["labels"] != cfg.ignore_index
                ).sum().item()

                # Optimizer step after accumulating grad_accum_steps batches
                if accum_step == cfg.grad_accum_steps - 1:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            get_lora_params(self.model), cfg.max_grad_norm
                        )

                    self._optimizer.step()
                    self._scheduler.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # ── Logging ────────────────────────────────────────
                    if global_step % cfg.log_steps == 0:
                        elapsed = time.perf_counter() - t_start
                        tok_per_sec = total_tokens / elapsed
                        avg_loss = running_loss / cfg.log_steps
                        lr = self._scheduler.get_last_lr()[0]

                        metrics = {
                            "loss": avg_loss,
                            "lr": lr,
                            "tokens_per_sec": tok_per_sec,
                            "step": global_step,
                            "epoch": epoch,
                        }
                        pbar.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            tok_s=f"{tok_per_sec:.0f}",
                            lr=f"{lr:.2e}",
                        )
                        if self._wandb:
                            self._wandb.log(metrics, step=global_step)

                        running_loss = 0.0

                    # ── Checkpoint ─────────────────────────────────────
                    if global_step % cfg.save_steps == 0:
                        eval_metrics = self.evaluate()
                        self._save_checkpoint(global_step)
                        if eval_metrics and self._wandb:
                            self._wandb.log(eval_metrics, step=global_step)

                    pbar.update(1)

        pbar.close()
        elapsed = time.perf_counter() - t_start

        # Final checkpoint
        self._save_checkpoint(global_step)
        final_eval = self.evaluate()

        summary = {
            "total_steps": global_step,
            "total_tokens": total_tokens,
            "tokens_per_sec": total_tokens / elapsed,
            "elapsed_sec": elapsed,
            **final_eval,
        }
        logger.info(
            f"\n{'Training complete':─^60}\n"
            f"  Total steps:    {global_step}\n"
            f"  Total tokens:   {total_tokens:,}\n"
            f"  Tokens/sec:     {summary['tokens_per_sec']:.1f}\n"
            f"  Elapsed:        {elapsed:.1f}s\n"
        )

        if self._wandb:
            self._wandb.finish()

        return summary