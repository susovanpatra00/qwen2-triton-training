"""
Model loader for Qwen2.5-7B (and other Qwen2.5 variants).

Handles:
- Standard float16 / bfloat16 loading
- 4-bit NF4 loading via bitsandbytes (base model only; LoRA layers stay bf16)
- Gradient checkpointing toggle
- Device placement

USAGE
-----
    from model.loader import load_qwen2

    model, tokenizer = load_qwen2(
        model_id="Qwen/Qwen2.5-7B",
        dtype=torch.bfloat16,
        load_in_4bit=True,          # optional
        gradient_checkpointing=True,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-72B",
    # Instruct variants
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
}


def load_qwen2(
    model_id: str = "Qwen/Qwen2.5-7B",
    *,
    dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device_map: str | dict = "auto",
    gradient_checkpointing: bool = True,
    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] = "sdpa",
    cache_dir: str | Path | None = None,
    trust_remote_code: bool = True,
) -> tuple:
    """
    Load a Qwen2.5 model and its tokenizer.

    Args:
        model_id:               HuggingFace model ID or local path.
        dtype:                  Compute dtype (bfloat16 recommended for Ampere+).
        load_in_4bit:           Enable 4-bit NF4 quantisation via bitsandbytes.
        load_in_8bit:           Enable 8-bit LLM.int8 quantisation.
        device_map:             Device placement — 'auto' shards across all GPUs.
        gradient_checkpointing: Trade compute for activation memory during training.
        attn_implementation:    Attention backend. Use 'flash_attention_2' if
                                flash-attn is installed (significant speedup).
        cache_dir:              Override HuggingFace cache directory.
        trust_remote_code:      Required for Qwen2.5 tokenizer.

    Returns:
        (model, tokenizer) tuple. Model is in train mode with requires_grad=False
        on all parameters (LoRA patching will unfreeze the adapters).
    """
    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot enable both 4-bit and 8-bit quantisation simultaneously.")

    logger.info(f"Loading {model_id} (dtype={dtype}, 4bit={load_in_4bit}, 8bit={load_in_8bit})")

    # ── Quantisation config ──────────────────────────────────────────────────
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # ── Load model ───────────────────────────────────────────────────────────
    model_kwargs: dict = dict(
        pretrained_model_name_or_path=model_id,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    if cache_dir:
        model_kwargs["cache_dir"] = str(cache_dir)
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # ── Gradient checkpointing ───────────────────────────────────────────────
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled")

    # Freeze all parameters; LoRA patching will unfreeze adapters
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        cache_dir=str(cache_dir) if cache_dir else None,
        padding_side="right",   # needed for training
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Loaded {model_id} — {n_params:.2f}B parameters")

    return model, tokenizer


def get_model_config(model_id: str) -> dict:
    """Return key architectural constants for a given Qwen2.5 model ID."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return {
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "num_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rms_norm_eps": cfg.rms_norm_eps,
    }