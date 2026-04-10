"""
Fused Cross-Entropy Loss — forward + backward in a single kernel.

WHY THIS EXISTS
---------------
For Qwen2.5-7B the vocabulary is ~150 000 tokens.  A single forward pass
with batch=4, seq=2048 produces a logits tensor of shape [8192, 150000].
At float32 that is ~4.9 GB just for *one* intermediate.  Standard PyTorch:

    1. Materialises logits            [M, V]  (written to HBM)
    2. Computes softmax               [M, V]  (second write)
    3. Saves softmax for backward     [M, V]  (stays resident)
    4. Backward reads saved softmax   [M, V]

Total HBM traffic ≈ 4 × M × V × element_size.

The fused kernel below never writes the full [M, V] softmax to HBM:

    Pass 1 (per-row): find max                  → 1 read of logits row
    Pass 2 (per-row): accumulate Σ exp, loss    → 1 read of logits row
    Pass 3 (per-row): write dlogits on-the-fly  → 1 read + 1 write

Total HBM traffic ≈ 3 × M × V × element_size (no saved softmax).

In practice the reduction from not saving the softmax activation saves
~1.5–2 GB on the above example, and removes one full round-trip to HBM
on every backward pass — that's where the tokens/sec improvement comes
from.

NUMERICS
--------
We use the standard log-sum-exp trick:

    log Z = max_i(x_i) + log Σ_i exp(x_i - max_i(x_i))
    loss  = log Z - x[label]

The gradient of the NLL loss w.r.t. logit i is:

    ∂L/∂x_i = (exp(x_i - log Z) − 𝟙[i == label]) × loss_scale

where loss_scale = 1/N for mean reduction.

USAGE
-----
    loss = fused_cross_entropy(logits, labels)          # mean
    loss = fused_cross_entropy(logits, labels, reduction="sum")
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _cross_entropy_fwd_kernel(
    logits_ptr,   # [M, V]  float16/bf16/float32
    labels_ptr,   # [M]     int64
    losses_ptr,   # [M]     float32  (output)
    lse_ptr,      # [M]     float32  log-sum-exp (saved for bwd)
    M,
    V,
    stride_m,     # stride along the M dimension of logits
    BLOCK_V: tl.constexpr,
):
    """Forward pass: compute per-token NLL loss and save log-sum-exp."""
    m = tl.program_id(0)
    row_start = logits_ptr + m * stride_m
    label = tl.load(labels_ptr + m)

    # ── Pass 1: row-wise maximum ─────────────────────────────────────────
    # Use Python float so accumulator stays scalar (not a [1] block tensor).
    m_val = float("-inf")
    for v_off in range(0, V, BLOCK_V):
        v_idx = v_off + tl.arange(0, BLOCK_V)
        mask = v_idx < V
        x = tl.load(row_start + v_idx, mask=mask, other=float("-inf")).to(tl.float32)
        m_val = tl.maximum(m_val, tl.max(x, axis=0))

    # ── Pass 2: log-sum-exp and per-token loss ───────────────────────────
    acc = 0.0  # scalar accumulator
    for v_off in range(0, V, BLOCK_V):
        v_idx = v_off + tl.arange(0, BLOCK_V)
        mask = v_idx < V
        x = tl.load(row_start + v_idx, mask=mask, other=float("-inf")).to(tl.float32)
        acc += tl.sum(tl.where(mask, tl.exp(x - m_val), 0.0), axis=0)

    lse = tl.log(acc) + m_val                        # log Z
    target_logit = tl.load(row_start + label).to(tl.float32)
    loss = lse - target_logit

    tl.store(losses_ptr + m, loss)
    tl.store(lse_ptr + m, lse)


@triton.jit
def _cross_entropy_bwd_kernel(
    dlogits_ptr,  # [M, V]  float32  (output gradients)
    logits_ptr,   # [M, V]  (original logits, read-only)
    labels_ptr,   # [M]     int64
    lse_ptr,      # [M]     float32  log-sum-exp from forward
    dloss_ptr,    # [M]     float32  upstream gradient (or scalar scale)
    M,
    V,
    stride_m,
    loss_scale,   # additional scalar (1/N for mean, 1 for sum)
    BLOCK_V: tl.constexpr,
):
    """
    Backward pass: compute dlogits without materialising full softmax.

        dlogit_i = (softmax_i − 𝟙[i==label]) × upstream_grad × loss_scale

    We compute softmax_i = exp(logit_i − lse) on-the-fly per chunk.
    """
    m = tl.program_id(0)
    row_logits = logits_ptr + m * stride_m
    row_dlogits = dlogits_ptr + m * stride_m

    label = tl.load(labels_ptr + m)
    lse = tl.load(lse_ptr + m)
    upstream = tl.load(dloss_ptr + m).to(tl.float32) * loss_scale

    for v_off in range(0, V, BLOCK_V):
        v_idx = v_off + tl.arange(0, BLOCK_V)
        mask = v_idx < V

        x = tl.load(row_logits + v_idx, mask=mask, other=float("-inf")).to(tl.float32)
        softmax_val = tl.exp(x - lse)                 # no full softmax written to HBM
        is_target = (v_idx == label).to(tl.float32)
        grad = (softmax_val - is_target) * upstream

        # Cast back to match the logits dtype
        tl.store(row_dlogits + v_idx, grad, mask=mask)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------
# Replace the entire _FusedCrossEntropy class with this:

class _FusedCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, loss_scale, valid_mask):
        M, V = logits.shape
        assert logits.is_contiguous()

        losses = torch.empty(M, dtype=torch.float32, device=logits.device)
        lse    = torch.empty(M, dtype=torch.float32, device=logits.device)

        BLOCK_V = triton.next_power_of_2(min(V, 4096))
        _cross_entropy_fwd_kernel[(M,)](
            logits, labels, losses, lse,
            M, V, logits.stride(0),
            BLOCK_V=BLOCK_V,
        )

        losses = losses * valid_mask.float()   # zero ignored positions

        ctx.save_for_backward(logits, labels, lse, valid_mask)
        ctx.loss_scale = loss_scale
        ctx.BLOCK_V = BLOCK_V
        return (losses * loss_scale).sum()

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, lse, valid_mask = ctx.saved_tensors
        M, V = logits.shape

        dloss   = grad_output.expand(M).contiguous()
        dlogits = torch.empty_like(logits, dtype=torch.float32)

        _cross_entropy_bwd_kernel[(M,)](
            dlogits, logits, labels, lse, dloss,
            M, V, logits.stride(0),
            loss_scale=ctx.loss_scale,
            BLOCK_V=ctx.BLOCK_V,
        )

        dlogits = dlogits * valid_mask.float().unsqueeze(1)  # zero ignored rows

        if logits.dtype != torch.float32:
            dlogits = dlogits.to(logits.dtype)

        return dlogits, None, None, None   # 4 Nones: logits, labels, loss_scale, valid_mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with a fused Triton kernel.

    Args:
        logits:      Float tensor of shape [N, V] or [B, T, V].
                     Will be reshaped to [M, V] internally.
        labels:      Long tensor of shape [N] or [B, T].
        reduction:   'mean' | 'sum' | 'none'
        ignore_index: Positions where labels == ignore_index are masked out.

    Returns:
        Scalar loss (or [M] if reduction='none').
    """
    orig_shape = logits.shape
    V = orig_shape[-1]
    logits_2d = logits.reshape(-1, V).contiguous()
    labels_1d = labels.reshape(-1)

    # Handle ignore_index by clamping labels and zeroing out those losses later.
    # We replace ignore positions with 0 (a valid token) so the kernel doesn't OOB,
    # then zero-out those positions' contribution.
    mask = labels_1d != ignore_index
    safe_labels = labels_1d.clone()
    safe_labels[~mask] = 0

    M = logits_2d.shape[0]
    loss_scale = 1.0 / max(mask.sum().item(), 1) if reduction == "mean" else 1.0

    raw_loss = _FusedCrossEntropy.apply(logits_2d, safe_labels, loss_scale, mask)

    if reduction == "none":
        # Recompute per-token losses (cheap: just the forward part)
        # Fall back to Python for now; fusing "none" reduction is straightforward
        # but makes the API more complex.
        with torch.no_grad():
            lp = torch.nn.functional.log_softmax(logits_2d.float(), dim=-1)
            per_tok = -lp.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
            per_tok = per_tok * mask.float()
        return per_tok.reshape(orig_shape[:-1])

    return raw_loss