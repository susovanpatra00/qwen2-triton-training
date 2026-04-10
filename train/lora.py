"""
LoRA — Low-Rank Adaptation from scratch.

Reference: Hu et al. 2021 (https://arxiv.org/abs/2106.09685)

THEORY
------
For a frozen pre-trained weight W ∈ R^{d_out × d_in}, LoRA adds a
low-rank update:

    W' = W + ΔW,   ΔW = (α/r) · B @ A

where:
    A ∈ R^{r × d_in}   initialised N(0, σ²)
    B ∈ R^{d_out × r}  initialised 0   (so ΔW = 0 at start)
    r                   rank (typically 8–64)
    α                   scaling factor (typically r or 2r)

During training only A and B are updated.  W is kept frozen.

IMPLEMENTATION NOTES
--------------------
- LoRALinear wraps an existing nn.Linear and shares its weight tensor
  (does not copy it).
- The bias, if present, is also frozen.
- We support optional dropout before the A projection (regularisation).
- merge() fuses ΔW into W for inference (removes the LoRA overhead).

TYPICAL USAGE
-------------
    config = LoRAConfig(r=16, alpha=32, target_modules=["q_proj", "v_proj"])
    model = apply_lora(model, config)

    # Inspect trainable parameters
    params = get_lora_params(model)
    optimizer = torch.optim.AdamW(params, lr=2e-4)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Hyperparameters for LoRA fine-tuning."""

    r: int = 16
    """Rank of the low-rank decomposition."""

    alpha: float = 32.0
    """Scaling factor.  Effective scale = alpha / r."""

    dropout: float = 0.05
    """Dropout probability applied before the A projection.  0 = disabled."""

    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    """
    Module name patterns to apply LoRA to.  Matched as substrings.
    Common choices for Qwen2.5:
        attention only:  ["q_proj", "k_proj", "v_proj", "o_proj"]
        + ffn gates:     ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    """

    bias: str = "none"
    """Which biases to train: 'none' | 'all' | 'lora_only'."""

    @property
    def scale(self) -> float:
        return self.alpha / self.r


# ---------------------------------------------------------------------------
# LoRALinear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    A nn.Linear layer augmented with a trainable low-rank adapter.

    The base weight is *frozen*.  Only lora_A, lora_B (and optionally bias)
    contribute gradients.
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.linear = linear                   # frozen base weight
        d_out, d_in = linear.weight.shape
        self.r = r
        self.scale = alpha / r

        device = linear.weight.device
        dtype  = linear.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=device, dtype=dtype))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze the base linear
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)

        # LoRA path: (x dropout) @ A^T @ B^T * scale
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + lora_out * self.scale

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """
        Fuse ΔW into the base weight and return a plain nn.Linear.
        After merging, there is zero inference overhead.
        """
        merged = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        delta_W = (self.lora_B @ self.lora_A).to(self.linear.weight.dtype)
        merged.weight = nn.Parameter(self.linear.weight + delta_W * self.scale)
        if self.linear.bias is not None:
            merged.bias = self.linear.bias
        return merged

    def extra_repr(self) -> str:
        d_out, d_in = self.linear.weight.shape
        return f"d_in={d_in}, d_out={d_out}, r={self.r}, scale={self.scale:.3f}"

    @property
    def weight(self):
        """Proxy so that code inspecting .weight still works."""
        return self.linear.weight


# ---------------------------------------------------------------------------
# Model-level helpers
# ---------------------------------------------------------------------------

def _module_matches(name: str, patterns: list[str]) -> bool:
    """Return True if any pattern is a substring of name."""
    return any(p in name for p in patterns)


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    *,
    verbose: bool = False,
) -> nn.Module:
    """
    Wrap all matching nn.Linear layers with LoRALinear in-place.

    Args:
        model:   Model to patch (modified in-place).
        config:  LoRAConfig specifying rank, alpha, targets.
        verbose: Log each patched module.

    Returns:
        The same model, with LoRA adapters attached.
    """
    replaced = 0
    skipped = 0

    for name, module in list(model.named_modules()):
        leaf_name = name.split(".")[-1]

        if not isinstance(module, nn.Linear):
            continue
        if not _module_matches(leaf_name, config.target_modules):
            skipped += 1
            continue

        # Navigate to parent
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        lora_layer = LoRALinear(
            module,
            r=config.r,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, parts[-1], lora_layer)
        replaced += 1

        if verbose:
            d_out, d_in = module.weight.shape
            n_lora_params = config.r * (d_in + d_out)
            logger.info(
                f"  ✓ LoRA [{name}]  shape=({d_out},{d_in})  "
                f"adapter_params={n_lora_params:,}"
            )

    # Handle bias training
    if config.bias == "all":
        for _, module in model.named_modules():
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.requires_grad_(True)
    elif config.bias == "lora_only":
        for _, module in model.named_modules():
            if isinstance(module, LoRALinear) and module.linear.bias is not None:
                module.linear.bias.requires_grad_(True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"LoRA applied: {replaced} layers wrapped, {skipped} skipped\n"
        f"  Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only the trainable LoRA parameters (A, B matrices)."""
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Extract only the LoRA adapter weights (small checkpoint).
    Load with model.load_state_dict(d, strict=False).
    """
    return {
        k: v
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Fuse all LoRA adapters into the base weights in-place.
    Call before export / inference to remove adapter overhead.
    """
    merged_count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        merged = module.merge()
        setattr(parent, parts[-1], merged)
        merged_count += 1

    logger.info(f"Merged {merged_count} LoRA adapters into base weights.")
    return model


def print_trainable_params(model: nn.Module) -> None:
    """Pretty-print a summary of trainable vs frozen parameters."""
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"\n{'Parameter summary':─^60}")
    print(f"  Total parameters:     {total:>15,}")
    print(f"  Trainable parameters: {trainable:>15,}")
    print(f"  Frozen parameters:    {total - trainable:>15,}")
    print(f"  Trainable fraction:   {100 * trainable / total:>14.4f}%")
    print("─" * 60)