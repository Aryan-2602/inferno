"""
bench_cache.py — Quantized KV cache vs baseline benchmark for Inferno.

Runs both the fp32 baseline and the INT8-quantized cache in the same script
to ensure a fair comparison. Measures tokens/sec, peak memory (MB), and
perplexity for each. Saves results to results/ as JSON and prints a comparison table.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from inferno.baseline import load_model, run_baseline
from inferno.cache import QuantizedDynamicCache, compute_perplexity
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


def run_benchmark() -> None:
    """
    Execute baseline and INT8-quantized runs on the same prompts and model,
    then print a comparison table and save JSON results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running benchmark on device: %s", device)

    model, tokenizer, device = load_model(device=device)

    # -----------------------------------------------------------------------
    # Baseline run
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # INT8 quantized run
    # -----------------------------------------------------------------------
    logger.info("=== INT8 quantized run ===")

    generated_texts_quant: list[str] = []
    total_new_tokens_quant = 0
    combined_cache = QuantizedDynamicCache()   # reused across prompts for stat tracking

    mem_tracker = GpuMemoryTracker(device)
    quant_start = wall_time()

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

            # Fresh cache per prompt — each sequence gets its own quantized KV cache.
            # We keep a reference to the last one for compression stats.
            prompt_cache = QuantizedDynamicCache()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    past_key_values=prompt_cache,
                )

            new_ids = output_ids[:, input_len:]
            total_new_tokens_quant += new_ids.numel()
            decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            generated_texts_quant.extend(decoded)

            # Accumulate compression ratios from this prompt's cache
            for layer in prompt_cache.layers:
                combined_cache._compression_ratios = getattr(combined_cache, '_compression_ratios', [])
            combined_cache = prompt_cache   # last prompt's cache for stats

    quant_total_time = wall_time() - quant_start
    quant_tps = total_new_tokens_quant / quant_total_time if quant_total_time > 0 else 0.0
    quant_peak_mb = mem_tracker.peak_mb
    mean_ratio = combined_cache.get_compression_stats()

    # Quantized perplexity — passes a fresh QuantizedDynamicCache per text
    quant_ppl = compute_perplexity(
        model, tokenizer, PERPLEXITY_TEXTS, device,
        cache=QuantizedDynamicCache(),
    )
    logger.info("Quantized perplexity: %.3f", quant_ppl)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "baseline": {
            "tokens_per_second": baseline_result.tokens_per_second,
            "peak_memory_mb": baseline_result.peak_memory_mb,
            "perplexity": baseline_ppl,
            "compression": 1.0,
        },
        "int8_quantized": {
            "tokens_per_second": quant_tps,
            "peak_memory_mb": quant_peak_mb,
            "perplexity": quant_ppl,
            "compression": mean_ratio,
        },
        "deltas": {
            "tokens_per_second": quant_tps - baseline_result.tokens_per_second,
            "peak_memory_mb": quant_peak_mb - baseline_result.peak_memory_mb,
            "perplexity": quant_ppl - baseline_ppl,
            "compression": mean_ratio - 1.0,
        },
    }
    path = save_results("bench_cache", results)
    logger.info("Results saved to %s", path)

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    tps_delta = quant_tps - baseline_result.tokens_per_second
    mem_delta = quant_peak_mb - baseline_result.peak_memory_mb
    ppl_delta = quant_ppl - baseline_ppl

    print()
    print("=" * 65)
    print(f"{'Metric':<20} | {'Baseline':>12} | {'INT8 Quant':>12} | {'Delta':>12}")
    print("-" * 65)
    print(f"{'Tokens/sec':<20} | {baseline_result.tokens_per_second:>12.2f} | {quant_tps:>12.2f} | {tps_delta:>+12.2f}")
    print(f"{'Peak memory MB':<20} | {baseline_result.peak_memory_mb:>12.1f} | {quant_peak_mb:>12.1f} | {mem_delta:>+12.1f}")
    print(f"{'Perplexity':<20} | {baseline_ppl:>12.3f} | {quant_ppl:>12.3f} | {ppl_delta:>+12.3f}")
    print(f"{'Compression':<20} | {'1x':>12} | {mean_ratio:>11.2f}x | {'':>12}")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
