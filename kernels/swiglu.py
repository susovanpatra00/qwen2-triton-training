"""
Fused SwiGLU — forward + backward Triton kernels.

Qwen2.5 uses SwiGLU as its FFN activation:

    FFN(x) = (SiLU(gate) ⊙ up) @ W_down

where gate = x @ W_gate,  up = x @ W_up.

This kernel fuses the element-wise part:

    out = SiLU(gate) * up

and its backward:

    ∂L/∂gate_i = ∂L/∂out_i  ×  ∂SiLU(gate_i)/∂gate_i  ×  up_i
    ∂L/∂up_i   = ∂L/∂out_i  ×  SiLU(gate_i)

where  SiLU(z) = z * σ(z)  and  ∂SiLU(z)/∂z = σ(z) * (1 + z*(1−σ(z))).

WHY FUSE
--------
Without fusion, PyTorch materialises intermediate tensors for gate, SiLU(gate),
and their product separately.  The fused kernel keeps everything in SRAM registers
for the element-wise computation, reducing HBM round-trips by ~2×.

USAGE
-----
    from kernels.swiglu import swiglu

    out = swiglu(gate, up)   # returns SiLU(gate) * up, differentiable
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

@triton.jit
def _silu(x):
    """SiLU(x) = x * σ(x) — implemented inline for register reuse."""
    return x * tl.sigmoid(x.to(tl.float32))


@triton.jit
def _silu_grad(x):
    """
    ∂SiLU(x)/∂x = σ(x) * (1 + x*(1−σ(x)))
    Numerically stable via sigmoid.
    """
    x_f = x.to(tl.float32)
    sig = tl.sigmoid(x_f)
    return sig * (1.0 + x_f * (1.0 - sig))


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _swiglu_fwd_kernel(
    Gate,   # [M, N]  input gate activations
    Up,     # [M, N]  input up-projection activations
    Out,    # [M, N]  output  SiLU(gate) * up
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    gate_row = Gate + m * stride_m
    up_row = Up + m * stride_m
    out_row = Out + m * stride_m

    for n_off in range(0, N, BLOCK_N):
        n_idx = n_off + tl.arange(0, BLOCK_N)
        mask = n_idx < N

        gate = tl.load(gate_row + n_idx, mask=mask, other=0.0)
        up = tl.load(up_row + n_idx, mask=mask, other=0.0)
        out = _silu(gate) * up.to(tl.float32)
        tl.store(out_row + n_idx, out.to(gate.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _swiglu_bwd_kernel(
    DGate,   # [M, N]  output: ∂L/∂gate
    DUp,     # [M, N]  output: ∂L/∂up
    Gate,    # [M, N]  saved gate
    Up,      # [M, N]  saved up
    DOut,    # [M, N]  upstream gradient ∂L/∂out
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    dgate_row = DGate + m * stride_m
    dup_row = DUp + m * stride_m
    gate_row = Gate + m * stride_m
    up_row = Up + m * stride_m
    dout_row = DOut + m * stride_m

    for n_off in range(0, N, BLOCK_N):
        n_idx = n_off + tl.arange(0, BLOCK_N)
        mask = n_idx < N

        gate = tl.load(gate_row + n_idx, mask=mask, other=0.0)
        up = tl.load(up_row + n_idx, mask=mask, other=0.0)
        dout = tl.load(dout_row + n_idx, mask=mask, other=0.0).to(tl.float32)

        gate_f = gate.to(tl.float32)
        up_f = up.to(tl.float32)

        silu_gate = _silu(gate_f)          # SiLU(gate)
        silu_d = _silu_grad(gate_f)        # d SiLU(gate) / d gate

        d_gate = dout * silu_d * up_f      # ∂L/∂gate
        d_up = dout * silu_gate            # ∂L/∂up

        tl.store(dgate_row + n_idx, d_gate.to(gate.dtype), mask=mask)
        tl.store(dup_row + n_idx, d_up.to(up.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class _SwiGLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        assert gate.shape == up.shape
        assert gate.is_contiguous() and up.is_contiguous()

        orig_shape = gate.shape
        N = orig_shape[-1]
        gate_2d = gate.reshape(-1, N)
        up_2d = up.reshape(-1, N)
        M = gate_2d.shape[0]

        out = torch.empty_like(gate_2d)
        BLOCK_N = triton.next_power_of_2(min(N, 2048))

        _swiglu_fwd_kernel[(M,)](
            gate_2d, up_2d, out,
            M, N, gate_2d.stride(0),
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(gate_2d, up_2d)
        ctx.orig_shape = orig_shape
        ctx.BLOCK_N = BLOCK_N
        return out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        gate_2d, up_2d = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        N = gate_2d.shape[1]
        M = gate_2d.shape[0]
        BLOCK_N = ctx.BLOCK_N

        dout_2d = dout.reshape(M, N).contiguous()
        dgate = torch.empty_like(gate_2d)
        dup = torch.empty_like(up_2d)

        _swiglu_bwd_kernel[(M,)](
            dgate, dup, gate_2d, up_2d, dout_2d,
            M, N, gate_2d.stride(0),
            BLOCK_N=BLOCK_N,
        )

        return dgate.reshape(orig_shape), dup.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU activation: SiLU(gate) * up.

    Args:
        gate: [..., N]  gate activations (output of W_gate)
        up:   [..., N]  up-projection activations (output of W_up)

    Returns:
        out: [..., N]  same shape as inputs
    """
    gate = gate.contiguous()
    up = up.contiguous()
    return _SwiGLU.apply(gate, up)


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class TritonSwiGLU(torch.nn.Module):
    """
    Drop-in SwiGLU module.

    Wraps the fused kernel so it can be used anywhere an activation function
    that consumes (gate, up) is expected.
    """

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return swiglu(gate, up)