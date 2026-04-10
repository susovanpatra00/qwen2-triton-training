"""
Model patching: swap PyTorch ops for Triton-fused equivalents.

Three patches are applied:

1. RMSNorm  → TritonRMSNorm
   Every Qwen2RMSNorm instance is replaced.  Weights are shared (not copied)
   so the patch is zero-cost in terms of memory.

2. SwiGLU   → TritonSwiGLU
   Qwen2MLP.forward() is monkey-patched to call the fused kernel instead of
   the two-step  act_fn(gate_proj(x)) * up_proj(x).

3. Cross-entropy (optional, applied inside the trainer)
   Not patched here — the trainer calls fused_cross_entropy directly since
   the loss is computed outside the model's forward().

USAGE
-----
    from model.loader import load_qwen2
    from model.patch  import patch_model

    model, tokenizer = load_qwen2("Qwen/Qwen2.5-7B")
    model = patch_model(model, verbose=True)
"""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from kernels.rms_norm import TritonRMSNorm
from kernels.swiglu import swiglu

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm patch
# ---------------------------------------------------------------------------

def _patch_rms_norms(model: nn.Module, verbose: bool = False) -> int:
    """
    Replace every Qwen2RMSNorm with TritonRMSNorm.
    Returns the number of modules replaced.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        cls_name = type(module).__name__
        if cls_name in ("Qwen2RMSNorm", "LlamaRMSNorm"):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if not parent_name else dict(model.named_modules())[parent_name]

            triton_norm = TritonRMSNorm(
                hidden_size=module.weight.shape[0],
                eps=module.variance_epsilon,
            )
            # Share the weight tensor so no extra memory is used
            triton_norm.weight = module.weight

            setattr(parent, child_name, triton_norm)
            replaced += 1
            if verbose:
                logger.info(f"  ✓ Replaced {cls_name} at '{name}'")

    return replaced


# ---------------------------------------------------------------------------
# SwiGLU patch
# ---------------------------------------------------------------------------

def _make_triton_mlp_forward(original_module: nn.Module):
    """
    Return a new forward function for a Qwen2MLP that calls the fused
    SwiGLU kernel instead of two separate ops.

    Original Qwen2MLP.forward:
        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    Patched version replaces  act_fn(gate_proj(x)) * up_proj(x)  with
    the single fused swiglu() call.
    """
    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = swiglu(gate, up)          # fused SiLU(gate) * up
        return self.down_proj(hidden)

    return _triton_forward


def _patch_swiglu(model: nn.Module, verbose: bool = False) -> int:
    """
    Monkey-patch every Qwen2MLP.forward to use the fused SwiGLU kernel.
    Returns the number of modules patched.
    """
    patched = 0
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen2MLP":
            module.forward = types.MethodType(_make_triton_mlp_forward(module), module)
            patched += 1
            if verbose:
                logger.info(f"  ✓ Patched Qwen2MLP at '{name}'")

    return patched


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def patch_model(
    model: "PreTrainedModel",
    *,
    patch_rms_norm: bool = True,
    patch_swiglu: bool = True,
    verbose: bool = False,
) -> "PreTrainedModel":
    """
    Apply Triton kernel patches to a loaded Qwen2.5 model in-place.

    Args:
        model:          The model returned by load_qwen2().
        patch_rms_norm: Replace all RMSNorm layers with Triton equivalents.
        patch_swiglu:   Fuse all MLP SwiGLU activations.
        verbose:        Log every substitution.

    Returns:
        The same model object (patched in-place, returned for convenience).
    """
    logger.info("Applying Triton kernel patches…")

    if patch_rms_norm:
        n = _patch_rms_norms(model, verbose=verbose)
        logger.info(f"  RMSNorm: replaced {n} modules")

    if patch_swiglu:
        n = _patch_swiglu(model, verbose=verbose)
        logger.info(f"  SwiGLU:  patched  {n} modules")

    logger.info("Patching complete.")
    return model


# ---------------------------------------------------------------------------
# Verification utility
# ---------------------------------------------------------------------------

def verify_patch(model: nn.Module, input_ids: torch.Tensor) -> bool:
    """
    Quick smoke test: run a forward pass and confirm no NaN/Inf outputs.
    Returns True if the patched model produces finite outputs.
    """
    model.eval()
    with torch.no_grad():
        try:
            out = model(input_ids, labels=input_ids)
            loss_ok = out.loss.isfinite().item()
            if not loss_ok:
                logger.warning("Patched model produced non-finite loss!")
            return loss_ok
        except Exception as exc:
            logger.error(f"Patched model forward pass failed: {exc}")
            return False