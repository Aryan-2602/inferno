"""
bench_cache.py — Quantized KV cache vs baseline benchmark for Inferno.

Runs baseline, INT8 per-tensor, and INT8 per-channel on the same prompts and
model to ensure a fair comparison. Measures tokens/sec, peak memory (MB), and
perplexity for each configuration. Saves results to results/ as JSON and prints
a three-way comparison table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from inferno.baseline import load_model, run_baseline
from inferno.cache import (
    QuantizedDynamicCache,
    QuantizedDynamicCachePerChannel,
    compute_perplexity,
)
from inferno.utils import GpuMemoryTracker, get_logger, save_results, wall_time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 64
BATCH_SIZE = 1
PERPLEXITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "In the beginning was the Word, and the Word was with God.",
    "To be or not to be, that is the question.",
    "It was the best of times, it was the worst of times.",
]
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

logger = get_logger(__name__)


def _run_quantized(
    model,
    tokenizer,
    device: torch.device,
    cache_cls: type,
    label: str,
) -> tuple[float, float, float, float]:
    """
    Run generate() over all PROMPTS using cache_cls as the KV cache, return
    (tokens_per_second, peak_memory_mb, perplexity, mean_compression_ratio).
    """
    logger.info("=== %s run ===", label)
    total_new_tokens = 0
    last_cache = cache_cls()

    mem_tracker = GpuMemoryTracker(device)
    start = wall_time()

    with mem_tracker:
        for batch_idx in range(0, len(PROMPTS), BATCH_SIZE):
            batch_prompts = PROMPTS[batch_idx : batch_idx + BATCH_SIZE]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            input_len = inputs["input_ids"].shape[1]

            prompt_cache = cache_cls()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    past_key_values=prompt_cache,
                )

            total_new_tokens += output_ids[:, input_len:].numel()
            last_cache = prompt_cache

    total_time = wall_time() - start
    tps = total_new_tokens / total_time if total_time > 0 else 0.0
    peak_mb = mem_tracker.peak_mb
    ratio = last_cache.get_compression_stats()

    ppl = compute_perplexity(model, tokenizer, PERPLEXITY_TEXTS, device, cache=cache_cls())
    logger.info("%s: %.2f tok/s | %.1f MB | ppl=%.3f | %.2fx", label, tps, peak_mb, ppl, ratio)
    return tps, peak_mb, ppl, ratio


def run_benchmark() -> None:
    """
    Run all three configurations and print a three-way comparison table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running benchmark on device: %s", device)

    model, tokenizer, device = load_model(device=device)

    # ---- Baseline ----
    logger.info("=== Baseline run ===")
    baseline_result = run_baseline(
        prompts=PROMPTS,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    baseline_ppl = compute_perplexity(model, tokenizer, PERPLEXITY_TEXTS, device)
    logger.info("Baseline perplexity: %.3f", baseline_ppl)

    # ---- INT8 per-tensor ----
    pt_tps, pt_mb, pt_ppl, pt_ratio = _run_quantized(
        model, tokenizer, device, QuantizedDynamicCache, "INT8 per-tensor"
    )

    # ---- INT8 per-channel ----
    pc_tps, pc_mb, pc_ppl, pc_ratio = _run_quantized(
        model, tokenizer, device, QuantizedDynamicCachePerChannel, "INT8 per-channel"
    )

    # ---- Save results ----
    results = {
        "baseline": {
            "tokens_per_second": baseline_result.tokens_per_second,
            "peak_memory_mb": baseline_result.peak_memory_mb,
            "perplexity": baseline_ppl,
            "compression": 1.0,
        },
        "int8_per_tensor": {
            "tokens_per_second": pt_tps,
            "peak_memory_mb": pt_mb,
            "perplexity": pt_ppl,
            "compression": pt_ratio,
        },
        "int8_per_channel": {
            "tokens_per_second": pc_tps,
            "peak_memory_mb": pc_mb,
            "perplexity": pc_ppl,
            "compression": pc_ratio,
        },
    }
    path = save_results("bench_cache", results)
    logger.info("Results saved to %s", path)

    # ---- Print comparison table ----
    b = baseline_result
    W = 14   # column width

    print()
    print("=" * 72)
    print(f"{'Metric':<20} | {'Baseline':>{W}} | {'Per-Tensor':>{W}} | {'Per-Channel':>{W}}")
    print("-" * 72)
    print(f"{'Tokens/sec':<20} | {b.tokens_per_second:>{W}.2f} | {pt_tps:>{W}.2f} | {pc_tps:>{W}.2f}")
    print(f"{'Peak memory MB':<20} | {b.peak_memory_mb:>{W}.1f} | {pt_mb:>{W}.1f} | {pc_mb:>{W}.1f}")
    print(f"{'Perplexity':<20} | {baseline_ppl:>{W}.3f} | {pt_ppl:>{W}.3f} | {pc_ppl:>{W}.3f}")
    print(f"{'Perplexity delta':<20} | {'—':>{W}} | {pt_ppl-baseline_ppl:>+{W}.3f} | {pc_ppl-baseline_ppl:>+{W}.3f}")
    print(f"{'Compression':<20} | {'1x':>{W}} | {pt_ratio:>{W-1}.2f}x | {pc_ratio:>{W-1}.2f}x")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
