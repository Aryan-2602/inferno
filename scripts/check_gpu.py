"""
check_gpu.py — GPU environment health check for Inferno Phase 3.

Run this first on the rented GPU machine before starting bench_gpu.py.
Verifies that CUDA is available, the model loads on GPU, quantized caches
work correctly on GPU, and reports peak GPU memory.

Usage:
    python scripts/check_gpu.py

Expected output on a healthy GPU environment:
    - CUDA available: True
    - Device name and VRAM
    - Model loaded successfully in bfloat16
    - Single forward pass completes, peak memory printed
    - Quantized generate() completes, compression ratio ~2x (bf16 input → int8)
    - All checks: PASSED
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from inferno.baseline import load_model
from inferno.cache import (
    QuantizedDynamicCache,
    QuantizedDynamicCachePerChannel,
    compute_perplexity,
)
from inferno.utils import GpuMemoryTracker, get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEALTH_PROMPT = "The quick brown fox jumps over the lazy dog"
MAX_NEW_TOKENS = 16
PPL_TEXTS = [
    "Artificial intelligence is transforming the world.",
    "The history of computing begins with mechanical devices.",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_gpu() -> None:
    """
    Run all environment health checks and print a summary.

    Exits with a non-zero status code if any check fails so this can be used
    as a gate in automated setup scripts.
    """
    failures: list[str] = []

    print()
    print("=" * 56)
    print("  INFERNO — GPU ENVIRONMENT HEALTH CHECK")
    print("=" * 56)

    # ---- CUDA availability ----
    print(f"\n[1] PyTorch version  : {torch.__version__}")
    print(f"    CUDA available   : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"    CUDA version     : {torch.version.cuda}")
        print(f"    Device count     : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            vram_gb = p.total_memory / (1024 ** 3)
            print(f"    Device {i}         : {p.name}  ({vram_gb:.1f} GB VRAM)")
    else:
        msg = "No CUDA device detected — Phase 3 GPU benchmarks require CUDA."
        print(f"    WARNING: {msg}")
        failures.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model load ----
    print(f"\n[2] Loading Qwen2.5-0.5B on {device} ...")
    try:
        model, tokenizer, device = load_model(device=device)
        param_m = sum(p.numel() for p in model.parameters()) // 1_000_000
        print(f"    Model loaded     : {param_m}M parameters, dtype={model.dtype}")
    except Exception as exc:
        msg = f"Model load failed: {exc}"
        print(f"    FAILED: {msg}")
        failures.append(msg)
        _print_summary(failures)
        sys.exit(1)

    # ---- Baseline forward pass ----
    print(f"\n[3] Single forward pass (greedy, {MAX_NEW_TOKENS} tokens) ...")
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        inputs = tokenizer(HEALTH_PROMPT, return_tensors="pt").to(device)
        with GpuMemoryTracker(device) as mem:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        output_text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"    Output           : {output_text!r}")
        print(f"    Peak memory      : {mem.peak_mb:.1f} MB")
        if device.type == "cuda" and mem.peak_mb < 100:
            msg = f"Peak GPU memory suspiciously low ({mem.peak_mb:.1f} MB) — check memory tracking."
            failures.append(msg)
            print(f"    WARNING: {msg}")
    except Exception as exc:
        msg = f"Forward pass failed: {exc}"
        print(f"    FAILED: {msg}")
        failures.append(msg)

    # ---- Quantized cache (per-tensor) ----
    print(f"\n[4] Quantized cache (INT8 per-tensor) ...")
    try:
        cache_pt = QuantizedDynamicCache()
        with torch.no_grad():
            out_q = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                past_key_values=cache_pt,
            )
        output_q = tokenizer.decode(out_q[0], skip_special_tokens=True)
        ratio = cache_pt.get_compression_stats()
        print(f"    Output           : {output_q!r}")
        print(f"    Compression      : {ratio:.2f}x")
        # GPU-NOTE: bf16 model on GPU gives ~2x compression (bf16→int8)
        # fp32 model on CPU gives ~4x compression (fp32→int8)
        expected_ratio = 2.0 if device.type == "cuda" else 4.0
        if abs(ratio - expected_ratio) > 0.5:
            msg = f"Unexpected compression ratio: {ratio:.2f}x (expected ~{expected_ratio:.1f}x)"
            failures.append(msg)
            print(f"    WARNING: {msg}")
    except Exception as exc:
        msg = f"Quantized cache (per-tensor) failed: {exc}"
        print(f"    FAILED: {msg}")
        failures.append(msg)

    # ---- Quantized cache (per-channel) ----
    print(f"\n[5] Quantized cache (INT8 per-channel) ...")
    try:
        cache_pc = QuantizedDynamicCachePerChannel()
        with torch.no_grad():
            out_pc = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                past_key_values=cache_pc,
            )
        output_pc = tokenizer.decode(out_pc[0], skip_special_tokens=True)
        ratio_pc = cache_pc.get_compression_stats()
        print(f"    Output           : {output_pc!r}")
        print(f"    Compression      : {ratio_pc:.2f}x")
    except Exception as exc:
        msg = f"Quantized cache (per-channel) failed: {exc}"
        print(f"    FAILED: {msg}")
        failures.append(msg)

    # ---- Perplexity (baseline) ----
    print(f"\n[6] Perplexity check (baseline) ...")
    try:
        ppl = compute_perplexity(model, tokenizer, PPL_TEXTS, device)
        print(f"    Baseline PPL     : {ppl:.3f}")
        if ppl > 100:
            msg = f"Baseline perplexity {ppl:.1f} is implausibly high — model may not have loaded correctly."
            failures.append(msg)
            print(f"    WARNING: {msg}")
    except Exception as exc:
        msg = f"Perplexity check failed: {exc}"
        print(f"    FAILED: {msg}")
        failures.append(msg)

    _print_summary(failures)
    if failures:
        sys.exit(1)


def _print_summary(failures: list[str]) -> None:
    """Print the final PASS / FAIL summary."""
    print()
    print("=" * 56)
    if not failures:
        print("  ALL CHECKS PASSED — environment ready for bench_gpu.py")
    else:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
    print("=" * 56)
    print()


if __name__ == "__main__":
    check_gpu()
