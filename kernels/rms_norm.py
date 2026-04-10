"""
Fused RMSNorm — forward + backward Triton kernels.

RMSNorm (Root Mean Square Layer Normalization, Zhang & Sennrich 2019):

    y = x / RMS(x) * w          where RMS(x) = sqrt(mean(x²) + ε)

Qwen2.5 uses RMSNorm before every attention block and FFN block, so this
op is called 2 × n_layers times per forward pass (64 times for 7B).

BACKWARD
--------
Let  ŷ = x / rms   (normalised, before weight multiplication)
     y = ŷ * w

∂L/∂w_j = Σ_m (∂L/∂y_{m,j}) * ŷ_{m,j}       (sum over batch/seq dim)

∂L/∂x_i = (1/rms) * [w_i * ∂L/∂y_i  −  ŷ_i * (1/N) * Σ_j w_j ∂L/∂y_j ŷ_j]

The fused kernel computes ∂L/∂x in one pass over the row (SRAM-resident),
and accumulates ∂L/∂w across rows atomically.

USAGE
-----
    from kernels.rms_norm import rms_norm

    y = rms_norm(x, weight, eps=1e-6)   # drop-in for LlamaRMSNorm
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _rms_norm_fwd_kernel(
    X,       # [M, N]  input
    W,       # [N]     scale weight
    Y,       # [M, N]  output
    Rstd,    # [M]     reciprocal std (saved for bwd)
    M, N,
    stride_m,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    row = X + m * stride_m
    out_row = Y + m * stride_m

    # ── Compute mean(x²) ────────────────────────────────────────────────
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for n_off in range(0, N, BLOCK_N):
        n_idx = n_off + tl.arange(0, BLOCK_N)
        mask = n_idx < N
        x = tl.load(row + n_idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(mask, x * x, 0.0)
    mean_sq = tl.sum(acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    tl.store(Rstd + m, rstd)

    # ── Normalise and scale ──────────────────────────────────────────────
    for n_off in range(0, N, BLOCK_N):
        n_idx = n_off + tl.arange(0, BLOCK_N)
        mask = n_idx < N
        # Load in original dtype first so we can cast back on store.
        x_orig = tl.load(row + n_idx, mask=mask, other=0.0)
        x = x_orig.to(tl.float32)
        w = tl.load(W + n_idx, mask=mask, other=1.0).to(tl.float32)
        y = x * rstd * w
        tl.store(out_row + n_idx, y.to(x_orig.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel — ∂L/∂x
# ---------------------------------------------------------------------------

@triton.jit
def _rms_norm_bwd_dx_kernel(
    DX,     # [M, N]  output: grad w.r.t. input
    DY,     # [M, N]  upstream gradient
    X,      # [M, N]  saved input
    W,      # [N]     weight
    Rstd,   # [M]     reciprocal RMS from forward
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    """Compute ∂L/∂x for one row.  ∂L/∂w is handled in a separate reduction."""
    m = tl.program_id(0)
    dx_row = DX + m * stride_m
    dy_row = DY + m * stride_m
    x_row = X + m * stride_m
    rstd = tl.load(Rstd + m)

    # ── Accumulate dot(w*dy, x̂) ─────────────────────────────────────────
    #    needed for the projection term in the gradient
    dot_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for n_off in range(0, N, BLOCK_N):
        n_idx = n_off + tl.arange(0, BLOCK_N)
        mask = n_idx < N
        x = tl.load(x_row + n_idx, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row + n_idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + n_idx, mask=mask, other=1.0).to(tl.float32)
        xhat = x * rstd
        dot_acc += tl.where(mask, w * dy * xhat, 0.0)
    dot = tl.sum(dot_acc, axis=0)  # scalar: Σ_j w_j dy_j x̂_j

    # ── Write ∂L/∂x_i = rstd * (w_i*dy_i − x̂_i * dot/N) ───────────────
    for n_off in range(0, N, BLOCK_N):
        n_idx = n_off + tl.arange(0, BLOCK_N)
        mask = n_idx < N
        x_orig = tl.load(x_row + n_idx, mask=mask, other=0.0)
        x = x_orig.to(tl.float32)
        dy = tl.load(dy_row + n_idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + n_idx, mask=mask, other=1.0).to(tl.float32)
        xhat = x * rstd
        dx = rstd * (w * dy - xhat * dot / N)
        tl.store(dx_row + n_idx, dx.to(x_orig.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel — ∂L/∂w  (cross-row reduction)
# ---------------------------------------------------------------------------

@triton.jit
def _rms_norm_bwd_dw_kernel(
    DW,     # [N]     output: grad w.r.t. weight
    DY,     # [M, N]  upstream gradient
    X,      # [M, N]  saved input
    Rstd,   # [M]     reciprocal RMS from forward
    M, N,
    stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Reduce ∂L/∂w_j = Σ_m (dy_{m,j} * x_{m,j} * rstd_m) over the M dimension.

    Each program handles a BLOCK_N-wide column slice and sums over all M rows.
    """
    n_off = tl.program_id(0) * BLOCK_N
    n_idx = n_off + tl.arange(0, BLOCK_N)
    n_mask = n_idx < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for m_off in range(0, M, BLOCK_M):
        m_idx = m_off + tl.arange(0, BLOCK_M)
        m_mask = m_idx < M

        # Load rstd for this block of rows
        rstd = tl.load(Rstd + m_idx, mask=m_mask, other=0.0)  # [BLOCK_M]

        # Load x and dy — shape [BLOCK_M, BLOCK_N]
        x = tl.load(
            X + m_idx[:, None] * stride_m + n_idx[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dy = tl.load(
            DY + m_idx[:, None] * stride_m + n_idx[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        xhat = x * rstd[:, None]
        acc += tl.sum(dy * xhat, axis=0)  # [BLOCK_N]

    tl.store(DW + n_idx, acc, mask=n_mask)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class _RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        M, N = x.shape
        assert x.is_contiguous()

        y = torch.empty_like(x)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        BLOCK_N = triton.next_power_of_2(min(N, 1024))
        _rms_norm_fwd_kernel[(M,)](
            x, weight, y, rstd,
            M, N,
            x.stride(0),
            eps=eps,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(x, weight, rstd)
        ctx.BLOCK_N = BLOCK_N
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, rstd = ctx.saved_tensors
        M, N = x.shape
        dy = dy.contiguous()

        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight, dtype=torch.float32)

        BLOCK_N = ctx.BLOCK_N
        BLOCK_M = max(1, triton.next_power_of_2(min(M, 32)))

        # ∂L/∂x (one program per row)
        _rms_norm_bwd_dx_kernel[(M,)](
            dx, dy, x, weight, rstd,
            M, N,
            x.stride(0),
            BLOCK_N=BLOCK_N,
        )

        # ∂L/∂w (one program per column block)
        n_programs = triton.cdiv(N, BLOCK_N)
        _rms_norm_bwd_dw_kernel[(n_programs,)](
            dw, dy, x, rstd,
            M, N,
            x.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        if weight.dtype != torch.float32:
            dw = dw.to(weight.dtype)

        return dx, dw, None   # no grad for eps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply RMSNorm with a fused Triton kernel.

    Args:
        x:       [..., N]  input tensor (will be reshaped to [M, N])
        weight:  [N]       learnable scale
        eps:     stability epsilon

    Returns:
        y: same shape as x
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    x_2d = x.reshape(-1, N).contiguous()

    y_2d = _RMSNorm.apply(x_2d, weight, eps)
    return y_2d.reshape(orig_shape)


# ---------------------------------------------------------------------------
# nn.Module wrapper (drop-in for transformers.models.qwen2.modeling_qwen2)
# ---------------------------------------------------------------------------

class TritonRMSNorm(torch.nn.Module):
    """Drop-in replacement for Qwen2RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.variance_epsilon)

    @classmethod
    def from_qwen_rms_norm(cls, module) -> "TritonRMSNorm":
        """Create from an existing Qwen2RMSNorm instance (copies weights)."""
        new = cls(module.weight.shape[0], eps=module.variance_epsilon)
        new.weight = module.weight  # share, not copy
        return new