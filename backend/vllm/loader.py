"""
vllm/loader.py
──────────────
Singleton loader for Qwen2.5-VL-7B-Instruct with 4-bit quantization.

The model is heavy (~6 GB VRAM with 4-bit quant), so we load it ONCE
the first time get_model_and_processor() is called and cache the result.
Subsequent calls return the already-loaded objects immediately.

This avoids reloading on every detection event.
"""

import logging
from typing import Tuple

import config

logger = logging.getLogger(__name__)

# Module-level singletons
_model = None
_processor = None


def get_model_and_processor() -> Tuple:
    """
    Return the loaded (model, processor) pair.
    Loads from HuggingFace on the first call; returns cached on subsequent calls.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded (e.g. no CUDA, insufficient VRAM).
    """
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    logger.info("Loading VLLM model: %s (4-bit quantized)", config.VLLM_MODEL_ID)

    try:
        import torch
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.VLLM_MODEL_ID,
            quantization_config=quant_config,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )

        _processor = AutoProcessor.from_pretrained(config.VLLM_MODEL_ID)

        vram_used_gb = torch.cuda.memory_allocated(0) / 1e9
        logger.info(
            "VLLM model loaded on cuda:0 | VRAM used: %.2f GB", vram_used_gb
        )

    except Exception as exc:
        logger.error("Failed to load VLLM model: %s", exc)
        raise RuntimeError(f"VLLM model load failed: {exc}") from exc

    return _model, _processor


def unload_model():
    """
    Explicitly free the model from VRAM.
    Useful if you want to reclaim GPU memory after processing is done.
    """
    global _model, _processor

    if _model is not None:
        import torch, gc
        del _model
        del _processor
        _model = None
        _processor = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("VLLM model unloaded from VRAM.")
