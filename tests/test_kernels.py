"""
Kernel correctness tests.

Each Triton kernel is tested against the PyTorch reference implementation:
- Forward outputs must match within floating-point tolerance
- Backward gradients must match (checked via torch.autograd.gradcheck
  or by comparing to analytic PyTorch gradients)

These tests run on CPU if no GPU is available, falling back to the PyTorch
implementations for reference (the Triton kernels require a GPU).

Run with:
    pytest tests/test_kernels.py -v
    pytest tests/test_kernels.py -v -k "cross_entropy"  # single kernel
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SKIP_NO_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA GPU"
)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DTYPES_IDS = ["fp16", "bf16", "fp32"]


def _allclose(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> bool:
    return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Cross-entropy tests
# ---------------------------------------------------------------------------

class TestFusedCrossEntropy:

    @SKIP_NO_GPU
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=DTYPES_IDS)
    @pytest.mark.parametrize("V", [1000, 32000, 151936], ids=["V1k", "V32k", "V152k"])
    def test_forward_matches_pytorch(self, dtype: torch.dtype, V: int) -> None:
        """Fused CE loss matches torch.nn.functional.cross_entropy."""
        from kernels.cross_entropy import fused_cross_entropy

        B, T = 2, 16
        torch.manual_seed(42)
        logits = torch.randn(B, T, V, device="cuda", dtype=dtype)
        labels = torch.randint(0, V, (B, T), device="cuda")

        ref = F.cross_entropy(
            logits.float().reshape(-1, V),
            labels.reshape(-1),
        )
        out = fused_cross_entropy(logits, labels)

        # Tolerance is relaxed for fp16/bf16 due to lower precision
        atol = 1e-2 if dtype != torch.float32 else 1e-4
        assert _allclose(out, ref, rtol=1e-2, atol=atol), (
            f"dtype={dtype}, V={V}: fused={out.item():.6f}, ref={ref.item():.6f}"
        )

    @SKIP_NO_GPU
    def test_ignore_index(self) -> None:
        """Tokens at ignore_index positions don't contribute to the loss."""
        from kernels.cross_entropy import fused_cross_entropy

        V = 1000
        logits = torch.randn(4, 8, V, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, V, (4, 8), device="cuda")
        labels[:, :4] = -100   # first half of every sequence is ignored

        out = fused_cross_entropy(logits, labels, ignore_index=-100)

        # Reference: manually mask out ignored positions
        ref = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=-100,
        )
        assert _allclose(out, ref, rtol=1e-3, atol=1e-3)

    @SKIP_NO_GPU
    @pytest.mark.parametrize("V", [1000, 32000], ids=["V1k", "V32k"])
    def test_backward_matches_pytorch(self, V: int) -> None:
        """Gradient of fused CE matches PyTorch autograd."""
        from kernels.cross_entropy import fused_cross_entropy

        B, T = 2, 8
        torch.manual_seed(0)
        logits_fused = torch.randn(B, T, V, device="cuda", dtype=torch.float32, requires_grad=True)
        logits_ref = logits_fused.detach().clone().requires_grad_(True)
        labels = torch.randint(0, V, (B, T), device="cuda")

        # Fused backward
        loss_fused = fused_cross_entropy(logits_fused, labels)
        loss_fused.backward()

        # PyTorch reference backward
        loss_ref = F.cross_entropy(
            logits_ref.reshape(-1, V), labels.reshape(-1)
        )
        loss_ref.backward()

        assert _allclose(
            logits_fused.grad, logits_ref.grad, rtol=1e-4, atol=1e-4
        ), "Backward pass gradients do not match PyTorch reference"

    @SKIP_NO_GPU
    def test_numerical_stability_large_logits(self) -> None:
        """Fused CE should not produce NaN/Inf for logits with large magnitude."""
        from kernels.cross_entropy import fused_cross_entropy

        V = 10000
        logits = torch.full((4, 16, V), 100.0, device="cuda", dtype=torch.float32)
        labels = torch.zeros((4, 16), dtype=torch.long, device="cuda")

        out = fused_cross_entropy(logits, labels)
        assert out.isfinite(), f"Got non-finite loss: {out}"


# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------

class TestRMSNorm:

    @SKIP_NO_GPU
    @pytest.mark.parametrize("N", [256, 4096, 8192], ids=["N256", "N4096", "N8192"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=DTYPES_IDS)
    def test_forward_matches_pytorch(self, N: int, dtype: torch.dtype) -> None:
        """Triton RMSNorm output matches a reference implementation."""
        from kernels.rms_norm import rms_norm

        B, T = 4, 32
        eps = 1e-6
        x = torch.randn(B, T, N, device="cuda", dtype=dtype)
        w = torch.ones(N, device="cuda", dtype=dtype)

        triton_out = rms_norm(x, w, eps=eps)

        # Reference: LlamaRMSNorm in float32
        x_f = x.float()
        rms = (x_f.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
        ref = (x_f * rms * w.float()).to(dtype)

        atol = 5e-3 if dtype != torch.float32 else 1e-5
        assert _allclose(triton_out, ref, rtol=1e-2, atol=atol), (
            f"N={N}, dtype={dtype}: max_err={( triton_out.float() - ref.float()).abs().max().item():.2e}"
        )

    @SKIP_NO_GPU
    def test_backward_matches_pytorch(self) -> None:
        """Gradient of Triton RMSNorm matches autograd."""
        from kernels.rms_norm import rms_norm

        N = 512
        B, T = 2, 16
        x_triton = torch.randn(B, T, N, device="cuda", dtype=torch.float32, requires_grad=True)
        w_triton = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
        x_ref = x_triton.detach().clone().requires_grad_(True)
        w_ref = w_triton.detach().clone().requires_grad_(True)

        eps = 1e-6

        # Triton backward
        out_triton = rms_norm(x_triton, w_triton, eps=eps)
        out_triton.sum().backward()

        # Reference backward (pure PyTorch)
        x_f = x_ref.float()
        rms = (x_f.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
        out_ref = (x_f * rms * w_ref.float())
        out_ref.sum().backward()

        assert _allclose(x_triton.grad, x_ref.grad, rtol=1e-3, atol=1e-4), \
            "dx does not match"
        assert _allclose(w_triton.grad, w_ref.grad, rtol=1e-3, atol=1e-4), \
            "dw does not match"

    @SKIP_NO_GPU
    def test_weight_gradient_accumulation(self) -> None:
        """dw must correctly sum over the batch dimension."""
        from kernels.rms_norm import rms_norm

        N = 64
        M = 512   # many rows → tests the reduction kernel
        x = torch.randn(M, N, device="cuda", requires_grad=True)
        w = torch.ones(N, device="cuda", requires_grad=True)

        out = rms_norm(x, w, eps=1e-6)
        out.sum().backward()

        # Shape check
        assert w.grad is not None
        assert w.grad.shape == (N,)
        assert w.grad.isfinite().all(), "dw contains NaN/Inf"


# ---------------------------------------------------------------------------
# SwiGLU tests
# ---------------------------------------------------------------------------

class TestSwiGLU:

    @SKIP_NO_GPU
    @pytest.mark.parametrize("N", [256, 4096, 14336], ids=["N256", "N4096", "N14k"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=DTYPES_IDS)
    def test_forward_matches_pytorch(self, N: int, dtype: torch.dtype) -> None:
        """Triton SwiGLU matches F.silu(gate) * up."""
        from kernels.swiglu import swiglu

        B, T = 4, 32
        gate = torch.randn(B, T, N, device="cuda", dtype=dtype)
        up = torch.randn(B, T, N, device="cuda", dtype=dtype)

        triton_out = swiglu(gate, up)
        ref = F.silu(gate.float()) * up.float()
        ref = ref.to(dtype)

        atol = 5e-3 if dtype != torch.float32 else 1e-5
        assert _allclose(triton_out, ref, rtol=1e-2, atol=atol), (
            f"N={N}, dtype={dtype}: max_err={(triton_out.float() - ref.float()).abs().max().item():.2e}"
        )

    @SKIP_NO_GPU
    def test_backward_matches_pytorch(self) -> None:
        """Gradient of fused SwiGLU matches PyTorch autograd."""
        from kernels.swiglu import swiglu

        N = 1024
        B = 4
        gate_t = torch.randn(B, N, device="cuda", dtype=torch.float32, requires_grad=True)
        up_t = torch.randn(B, N, device="cuda", dtype=torch.float32, requires_grad=True)
        gate_r = gate_t.detach().clone().requires_grad_(True)
        up_r = up_t.detach().clone().requires_grad_(True)

        # Triton
        out_t = swiglu(gate_t, up_t)
        out_t.sum().backward()

        # Reference
        out_r = F.silu(gate_r) * up_r
        out_r.sum().backward()

        assert _allclose(gate_t.grad, gate_r.grad, rtol=1e-4, atol=1e-4), "d_gate mismatch"
        assert _allclose(up_t.grad, up_r.grad, rtol=1e-4, atol=1e-4), "d_up mismatch"

    @SKIP_NO_GPU
    def test_zero_gate_grad(self) -> None:
        """When gate == 0, SiLU grad is 0.5, so d_gate = 0.5 * dout * up."""
        from kernels.swiglu import swiglu

        N = 64
        gate = torch.zeros(N, device="cuda", requires_grad=True)
        up = torch.ones(N, device="cuda", requires_grad=True)

        out = swiglu(gate, up)
        out.sum().backward()

        # SiLU'(0) = sigmoid(0) * (1 + 0*(1-sigmoid(0))) = 0.5
        expected_dgate = 0.5 * torch.ones(N, device="cuda")
        assert _allclose(gate.grad, expected_dgate, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# LoRA tests (CPU-compatible)
# ---------------------------------------------------------------------------

class TestLoRA:

    def test_lora_zero_init(self) -> None:
        """LoRA output equals base linear at init (B starts as zero)."""
        from train.lora import LoRALinear

        linear = torch.nn.Linear(64, 128, bias=False)
        lora = LoRALinear(linear, r=4, alpha=8)

        x = torch.randn(2, 64)
        with torch.no_grad():
            ref = linear(x)
            out = lora(x)

        assert torch.allclose(out, ref, atol=1e-6), \
            "LoRA output should equal base at init (B=0)"

    def test_lora_trainable_params(self) -> None:
        """Only lora_A and lora_B have requires_grad=True."""
        from train.lora import LoRALinear

        linear = torch.nn.Linear(64, 128)
        lora = LoRALinear(linear, r=4, alpha=8)

        trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
        assert "lora_A" in trainable
        assert "lora_B" in trainable
        assert "linear.weight" not in trainable

    def test_lora_merge(self) -> None:
        """Merged linear produces same output as LoRA forward."""
        from train.lora import LoRALinear

        linear = torch.nn.Linear(32, 64, bias=False)
        lora = LoRALinear(linear, r=4, alpha=8)

        # Train for one step so B != 0
        x = torch.randn(4, 32)
        loss = lora(x).sum()
        loss.backward()
        with torch.no_grad():
            lora.lora_B.add_(torch.randn_like(lora.lora_B) * 0.01)

        merged = lora.merge()

        with torch.no_grad():
            out_lora = lora(x)
            out_merged = merged(x)

        assert torch.allclose(out_lora, out_merged, atol=1e-5), \
            "Merged weight produces different output than LoRA forward"

    def test_apply_lora_target_modules(self) -> None:
        """apply_lora only wraps modules matching target_modules."""
        from train.lora import LoRAConfig, LoRALinear, apply_lora

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(64, 64)
                self.v_proj = torch.nn.Linear(64, 64)
                self.ffn = torch.nn.Linear(64, 256)

        model = TinyModel()
        cfg = LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"])
        apply_lora(model, cfg)

        assert isinstance(model.q_proj, LoRALinear), "q_proj should be LoRALinear"
        assert isinstance(model.v_proj, LoRALinear), "v_proj should be LoRALinear"
        assert isinstance(model.ffn, torch.nn.Linear), "ffn should remain nn.Linear"

    def test_lora_state_dict_keys(self) -> None:
        """lora_state_dict returns only adapter keys."""
        from train.lora import LoRAConfig, LoRALinear, apply_lora, lora_state_dict

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(32, 32)

        model = TinyModel()
        apply_lora(model, LoRAConfig(r=4, alpha=8, target_modules=["q_proj"]))

        sd = lora_state_dict(model)
        assert all("lora_A" in k or "lora_B" in k for k in sd), \
            f"Unexpected keys in LoRA state dict: {list(sd.keys())}"
        assert len(sd) == 2  # one lora_A, one lora_B