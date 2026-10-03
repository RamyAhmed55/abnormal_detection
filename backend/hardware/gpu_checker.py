"""
hardware/gpu_checker.py
───────────────────────
Checks GPU availability and free VRAM, then decides whether to use
the local Qwen VLLM model or route to the Gemini API.

Decision tree
─────────────
  CUDA available?
    └─ NO  → route = "gemini_api"
    └─ YES → free VRAM >= VLLM_MIN_VRAM_GB?
                └─ NO  → route = "gemini_api"
                └─ YES → route = "local_vllm"

The result is computed once at startup and cached.
"""

import logging
from typing import Literal

import config

logger = logging.getLogger(__name__)

# Cached result (set once on first call to get_routing_decision)
_cached_route: str = None


RouteType = Literal["local_vllm", "gemini_api"]


def get_routing_decision(force: str = None) -> RouteType:
    """
    Return 'local_vllm' or 'gemini_api'.

    Parameters
    ----------
    force : str | None
        If set to 'local_vllm' or 'gemini_api', skip hardware checks
        and return that value directly (useful for testing/CLI override).

    Returns
    -------
    RouteType
    """
    global _cached_route

    if force in ("local_vllm", "gemini_api"):
        logger.info("Route forced by caller: %s", force)
        _cached_route = force
        return _cached_route

    if _cached_route is not None:
        return _cached_route

    _cached_route = _detect_route()
    return _cached_route


def _detect_route() -> RouteType:
    """Perform the actual GPU check and return the appropriate route."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed — cannot use local VLLM. Routing to Gemini API.")
        return "gemini_api"

    if not torch.cuda.is_available():
        logger.info("No CUDA GPU detected → routing to Gemini API.")
        return "gemini_api"

    # GPU found — check free VRAM
    device = torch.device("cuda:0")
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    allocated_gb = torch.cuda.memory_allocated(device) / 1e9
    free_vram_gb = total_vram_gb - allocated_gb

    gpu_name = torch.cuda.get_device_name(device)

    logger.info(
        "GPU detected: %s | Total VRAM: %.2f GB | Free: %.2f GB",
        gpu_name, total_vram_gb, free_vram_gb,
    )

    if free_vram_gb < config.VLLM_MIN_VRAM_GB:
        logger.warning(
            "Free VRAM %.2f GB < required %.2f GB → routing to Gemini API.",
            free_vram_gb, config.VLLM_MIN_VRAM_GB,
        )
        return "gemini_api"

    logger.info(
        "Sufficient VRAM available (%.2f GB free) → using local VLLM (Qwen2.5-VL-7B).",
        free_vram_gb,
    )
    return "local_vllm"


def print_hardware_summary():
    """Print a human-readable summary of the hardware decision at startup."""
    route = get_routing_decision()
    print("\n" + "─" * 55)
    print("  🖥  Hardware Check")
    print("─" * 55)

    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            name = torch.cuda.get_device_name(device)
            total = torch.cuda.get_device_properties(device).total_memory / 1e9
            free = total - torch.cuda.memory_allocated(device) / 1e9
            print(f"  GPU   : {name}")
            print(f"  VRAM  : {total:.1f} GB total | {free:.1f} GB free")
        else:
            print("  GPU   : Not available (CPU only)")
    except Exception:
        print("  GPU   : Unknown (torch not installed or error)")

    symbol = "✅" if route == "local_vllm" else "☁️"
    label  = "Local Qwen2.5-VL-7B (VLLM)" if route == "local_vllm" else "Gemini API (cloud)"
    print(f"  Route : {symbol} {label}")
    print("─" * 55 + "\n")
