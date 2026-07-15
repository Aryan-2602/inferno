"""
bench_gpu.py — GPU validation benchmark for Inferno Phase 3.

Runs on the rented GPU machine (RunPod A10/4090 or similar).
Do NOT run on CPU — it will be very slow and the memory numbers will be wrong.

Before running:
    python scripts/check_gpu.py   # must show ALL CHECKS PASSED

What this script measures:
  Part A — KV cache comparison (Baseline | INT8 Per-Tensor | INT8 Per-Channel)
    - Tokens/sec
    - Peak GPU memory MB (torch.cuda.max_memory_allocated)
    - Perplexity on 20 wikitext-2 excerpts
    - Time to first token (TTFT ms)

  Part B — Static vs Continuous batching
    - Same fixed workload (parity-checked)
    - Tokens/sec, mean latency ms, max latency ms

Usage:
    source .venv/bin/activate
    python benchmarks/bench_gpu.py

Results saved to results/bench_gpu_{timestamp}.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

if not torch.cuda.is_available():
    print("ERROR: No CUDA device found. bench_gpu.py requires a GPU.")
    print("Run scripts/check_gpu.py first to diagnose the environment.")
    sys.exit(1)

from datasets import load_dataset

from inferno.baseline import load_model, run_baseline
from inferno.cache import (
    QuantizedDynamicCache,
    QuantizedDynamicCachePerChannel,
    compute_perplexity,
)
from inferno.engine import ContinuousBatchingEngine, SchedulerConfig
from inferno.utils import GpuMemoryTracker, get_logger, save_results, wall_time
from benchmarks.bench_batching import build_workload, run_static_batching, run_continuous_batching

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 64
BATCH_SIZE = 1
WIKITEXT_NUM_SAMPLES = 20       # number of excerpts from wikitext-2 test set
WIKITEXT_MIN_CHARS = 100        # skip very short lines
PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "What are the main causes of the French Revolution?",
    "Describe how a neural network learns from data.",
    "What is the difference between machine learning and deep learning?",
    "How does photosynthesis work?",
    "Explain quantum entanglement to a five-year-old.",
    "What are the key events of World War II?",
    "How does the human immune system fight viruses?",
    "What is the significance of the Turing test?",
    "Describe the water cycle and its importance.",
]
STATIC_BATCH_SIZE = 4

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wikitext perplexity corpus
# ---------------------------------------------------------------------------

def load_wikitext_samples() -> list[str]:
    """
    Load WIKITEXT_NUM_SAMPLES excerpts from the wikitext-2-raw-v1 test split.

    Filters out blank lines and headings (lines starting with '=') to get
    plain prose passages that give a meaningful perplexity reading.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    samples: list[str] = []
    for row in ds:
        text = row["text"].strip()
        if len(text) >= WIKITEXT_MIN_CHARS and not text.startswith("="):
            samples.append(text)
        if len(samples) >= WIKITEXT_NUM_SAMPLES:
            break
    if len(samples) < WIKITEXT_NUM_SAMPLES:
        logger.warning(
            "Only %d wikitext samples available (needed %d)", len(samples), WIKITEXT_NUM_SAMPLES
        )
    return samples


# ---------------------------------------------------------------------------
# TTFT measurement helper
# ---------------------------------------------------------------------------

def measure_ttft(model, tokenizer, device: torch.device, prompt: str) -> float:
    """
    Measure time-to-first-token (TTFT) in milliseconds.

    TTFT = wall time for a single prefill forward pass on one prompt.
    Excludes tokenization and any KV cache setup so the number reflects
    pure attention + MLP latency for the prompt length.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    if device.type == "cuda":
        # Warm up to avoid one-time CUDA initialization cost in the measurement.
        with torch.no_grad():
            _ = model(**enc)
        torch.cuda.synchronize(device)

    t0 = wall_time()
    with torch.no_grad():
        _ = model(**enc)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (wall_time() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Part A — KV cache comparison
# ---------------------------------------------------------------------------

def run_cache_benchmark(
    model,
    tokenizer,
    device: torch.device,
    wikitext_samples: list[str],
) -> dict:
    """
    Run baseline, INT8 per-tensor, and INT8 per-channel on the same 10 prompts.

    Returns a dict with results for all three configurations.
    """

    def _run_config(label: str, cache_cls=None) -> dict:
        logger.info("=== %s ===", label)

        # Reset GPU memory stats before each run so peak reflects only this config.
        torch.cuda.reset_peak_memory_stats(device)

        total_new_tokens = 0
        mem_tracker = GpuMemoryTracker(device)
        t_start = wall_time()

        with mem_tracker:
            for batch_idx in range(0, len(PROMPTS), BATCH_SIZE):
                batch = PROMPTS[batch_idx : batch_idx + BATCH_SIZE]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                input_len = inputs["input_ids"].shape[1]

                gen_kwargs: dict = dict(
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if cache_cls is not None:
                    gen_kwargs["past_key_values"] = cache_cls()

                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)

                total_new_tokens += out[:, input_len:].numel()

        total_time = wall_time() - t_start
        tps = total_new_tokens / total_time if total_time > 0 else 0.0
        peak_mb = mem_tracker.peak_mb

        # Perplexity on wikitext (more standard than custom texts)
        ppl_cache = cache_cls() if cache_cls is not None else None
        ppl = compute_perplexity(model, tokenizer, wikitext_samples, device, cache=ppl_cache)

        # TTFT on a representative medium-length prompt
        ttft_ms = measure_ttft(model, tokenizer, device, PROMPTS[2])

        # Compression ratio (only meaningful for quantized configs)
        if cache_cls is not None:
            probe_cache = cache_cls()
            with torch.no_grad():
                model.generate(
                    **tokenizer(PROMPTS[0], return_tensors="pt").to(device),
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    past_key_values=probe_cache,
                )
            compression = probe_cache.get_compression_stats()
        else:
            compression = 1.0

        logger.info(
            "%s: %.1f tok/s | %.1f MB | ppl=%.3f | ttft=%.1f ms | %.2fx",
            label, tps, peak_mb, ppl, ttft_ms, compression,
        )
        return {
            "tokens_per_second": tps,
            "peak_memory_mb": peak_mb,
            "perplexity": ppl,
            "ttft_ms": ttft_ms,
            "compression": compression,
        }

    baseline = _run_config("Baseline", cache_cls=None)
    per_tensor = _run_config("INT8 Per-Tensor", cache_cls=QuantizedDynamicCache)
    per_channel = _run_config("INT8 Per-Channel", cache_cls=QuantizedDynamicCachePerChannel)

    return {
        "baseline": baseline,
        "int8_per_tensor": per_tensor,
        "int8_per_channel": per_channel,
    }


# ---------------------------------------------------------------------------
# Part B — Static vs Continuous batching
# ---------------------------------------------------------------------------

def run_batching_benchmark(model, tokenizer, device: torch.device) -> dict:
    """
    Static vs continuous batching on the canonical heterogeneous workload.

    Uses build_workload() so both strategies process identical request lists.
    Asserts total token parity before returning.
    """
    static_requests = build_workload()
    continuous_requests = build_workload()

    logger.info("=== Static batching ===")
    static = run_static_batching(static_requests, model, tokenizer, device)

    logger.info("=== Continuous batching ===")
    continuous = run_continuous_batching(continuous_requests, model, tokenizer, device)

    if static["total_tokens"] != continuous["total_tokens"]:
        raise ValueError(
            f"Workload parity violation: static={static['total_tokens']} tokens "
            f"vs continuous={continuous['total_tokens']} tokens."
        )

    return {"static_batching": static, "continuous_batching": continuous}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    """Run the full GPU benchmark suite and print both comparison tables."""
    device = torch.device("cuda")
    logger.info("GPU benchmark starting on: %s", torch.cuda.get_device_name(device))

    model, tokenizer, device = load_model(device=device)

    logger.info("Loading wikitext-2 test samples ...")
    wikitext_samples = load_wikitext_samples()
    logger.info("Loaded %d wikitext samples for perplexity.", len(wikitext_samples))

    # ---- Part A ----
    cache_results = run_cache_benchmark(model, tokenizer, device, wikitext_samples)

    # ---- Part B ----
    batching_results = run_batching_benchmark(model, tokenizer, device)

    # ---- Save ----
    all_results = {
        "device": torch.cuda.get_device_name(device),
        "cache_benchmark": cache_results,
        "batching_benchmark": batching_results,
    }
    path = save_results("bench_gpu", all_results)
    logger.info("Results saved to %s", path)

    # ---- Print Part A table ----
    b = cache_results["baseline"]
    pt = cache_results["int8_per_tensor"]
    pc = cache_results["int8_per_channel"]
    W = 14

    print()
    print("=" * 74)
    print("  Part A — KV Cache Quantization (GPU)")
    print("=" * 74)
    print(f"{'Metric':<20} | {'Baseline':>{W}} | {'Per-Tensor':>{W}} | {'Per-Channel':>{W}}")
    print("-" * 74)
    print(f"{'Tokens/sec':<20} | {b['tokens_per_second']:>{W}.1f} | {pt['tokens_per_second']:>{W}.1f} | {pc['tokens_per_second']:>{W}.1f}")
    print(f"{'Peak memory MB':<20} | {b['peak_memory_mb']:>{W}.1f} | {pt['peak_memory_mb']:>{W}.1f} | {pc['peak_memory_mb']:>{W}.1f}")
    print(f"{'Perplexity':<20} | {b['perplexity']:>{W}.3f} | {pt['perplexity']:>{W}.3f} | {pc['perplexity']:>{W}.3f}")
    print(f"{'Perplexity delta':<20} | {'—':>{W}} | {pt['perplexity']-b['perplexity']:>+{W}.3f} | {pc['perplexity']-b['perplexity']:>+{W}.3f}")
    print(f"{'Compression':<20} | {'1x':>{W}} | {pt['compression']:>{W-1}.2f}x | {pc['compression']:>{W-1}.2f}x")
    print(f"{'TTFT ms':<20} | {b['ttft_ms']:>{W}.1f} | {pt['ttft_ms']:>{W}.1f} | {pc['ttft_ms']:>{W}.1f}")
    print("=" * 74)

    # ---- Print Part B table ----
    s = batching_results["static_batching"]
    c = batching_results["continuous_batching"]

    print()
    print("=" * 58)
    print("  Part B — Batching Strategy (GPU)")
    print("=" * 58)
    print(f"{'Metric':<22} | {'Static':>{W}} | {'Continuous':>{W}}")
    print("-" * 58)
    print(f"{'Tokens/sec':<22} | {s['tokens_per_second']:>{W}.1f} | {c['tokens_per_second']:>{W}.1f}")
    print(f"{'Mean latency ms':<22} | {s['mean_latency_ms']:>{W}.0f} | {c['mean_latency_ms']:>{W}.0f}")
    print(f"{'p50 latency ms':<22} | {s['p50_latency_ms']:>{W}.0f} | {c['p50_latency_ms']:>{W}.0f}")
    print(f"{'p95 latency ms':<22} | {s['p95_latency_ms']:>{W}.0f} | {c['p95_latency_ms']:>{W}.0f}")
    print(f"{'p99 latency ms':<22} | {s['p99_latency_ms']:>{W}.0f} | {c['p99_latency_ms']:>{W}.0f}")
    print(f"{'Max latency ms':<22} | {s['max_latency_ms']:>{W}.0f} | {c['max_latency_ms']:>{W}.0f}")
    print(f"{'Total tokens':<22} | {s['total_tokens']:>{W}d} | {c['total_tokens']:>{W}d}")
    print("=" * 58)

    mean_delta = c["mean_latency_ms"] - s["mean_latency_ms"]
    tps_delta_pct = (c["tokens_per_second"] - s["tokens_per_second"]) / s["tokens_per_second"] * 100
    print(f"\nMean latency delta : {mean_delta:+.0f} ms")
    print(f"Throughput delta   : {tps_delta_pct:+.1f}%")


if __name__ == "__main__":
    run_benchmark()
