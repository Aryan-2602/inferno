"""
bench_batching.py — Continuous batching throughput benchmark for Inferno.

Compares two scheduling strategies on the same workload:

  Static batching  — pad all prompts to the same length, call model.generate()
                     once for the whole batch. All sequences finish at the same time
                     (determined by the longest sequence in the batch).

  Continuous batching — each request has its own max_new_tokens; short sequences
                        finish early and free their slot for new work.

The key insight continuous batching exploits: shorter sequences don't wait for
the longest one. On a heterogeneous workload (mixed lengths), this gives lower
mean latency while preserving throughput.

Saves results to results/ as JSON and prints a comparison table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from inferno.baseline import load_model
from inferno.engine import ContinuousBatchingEngine, Request, SchedulerConfig
from inferno.utils import get_logger, save_results, wall_time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Heterogeneous workload: different max_new_tokens per prompt to make the
# scheduling difference visible.
WORKLOAD: list[tuple[str, int]] = [
    ("Hi", 8),
    ("The capital of France is", 8),
    ("Explain gravity briefly", 16),
    ("What is 2+2?", 8),
    ("Tell me about machine learning in detail", 32),
    ("Name three planets", 8),
    ("Summarise quantum mechanics", 32),
    ("What color is the sky?", 8),
]

STATIC_BATCH_SIZE = 4   # number of sequences processed together in static batching
STATIC_MAX_NEW_TOKENS = max(tok for _, tok in WORKLOAD)  # pad all to worst case

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Static batching baseline
# ---------------------------------------------------------------------------

def run_static_batching(model, tokenizer, device: torch.device) -> dict:
    """
    Naïve static batching: group all prompts into one padded batch and call
    model.generate() once per batch with a fixed token budget equal to the
    maximum across the whole workload.

    Every sequence in the batch burns STATIC_MAX_NEW_TOKENS decode steps,
    even short ones that could have finished much earlier. This wastes compute
    and inflates mean latency.
    """
    prompts = [p for p, _ in WORKLOAD]
    n = len(prompts)

    # Process in a single batch (or in sub-batches of STATIC_BATCH_SIZE)
    total_tokens = 0
    latencies_ms: list[float] = []

    start_all = wall_time()

    for batch_start in range(0, n, STATIC_BATCH_SIZE):
        batch_prompts = prompts[batch_start : batch_start + STATIC_BATCH_SIZE]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        input_len = enc["input_ids"].shape[1]

        t0 = wall_time()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=STATIC_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed_ms = (wall_time() - t0) * 1000.0

        new_tokens = out[:, input_len:].numel()
        total_tokens += new_tokens

        # Every sequence in the batch waits the same time (static scheduling)
        for _ in batch_prompts:
            latencies_ms.append(elapsed_ms)

    total_time = wall_time() - start_all
    tps = total_tokens / total_time if total_time > 0 else 0.0
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    max_lat = max(latencies_ms)

    logger.info(
        "Static batching: %.2f tok/s | mean lat=%.0f ms | max lat=%.0f ms",
        tps, mean_lat, max_lat,
    )
    return {
        "tokens_per_second": tps,
        "mean_latency_ms": mean_lat,
        "max_latency_ms": max_lat,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Continuous batching
# ---------------------------------------------------------------------------

def run_continuous_batching(model, tokenizer, device: torch.device) -> dict:
    """
    Continuous batching via ContinuousBatchingEngine: each request carries its
    own max_new_tokens budget. Short sequences exit early, freeing their slot
    for the next waiting request without stalling longer-running ones.
    """
    config = SchedulerConfig(max_batch_size=STATIC_BATCH_SIZE, max_prefill_chunk_size=512)
    engine = ContinuousBatchingEngine(model=model, tokenizer=tokenizer, config=config, device=device)

    requests: list[Request] = []
    for prompt, max_tok in WORKLOAD:
        req = Request(prompt=prompt, max_new_tokens=max_tok)
        requests.append(req)
        engine.submit(req)

    start_all = wall_time()
    completed = engine.run_until_complete()
    total_time = wall_time() - start_all

    latencies_ms = [cr.latency_ms for cr in completed]
    total_tokens = sum(cr.tokens_generated for cr in completed)
    tps = total_tokens / total_time if total_time > 0 else 0.0
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    max_lat = max(latencies_ms)

    logger.info(
        "Continuous batching: %.2f tok/s | mean lat=%.0f ms | max lat=%.0f ms",
        tps, mean_lat, max_lat,
    )
    return {
        "tokens_per_second": tps,
        "mean_latency_ms": mean_lat,
        "max_latency_ms": max_lat,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    """
    Run static and continuous batching on the same workload and print a
    comparison table showing tokens/sec, mean latency, and max latency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running batching benchmark on device: %s", device)
    model, tokenizer, device = load_model(device=device)

    logger.info("=== Static batching ===")
    static = run_static_batching(model, tokenizer, device)

    logger.info("=== Continuous batching ===")
    continuous = run_continuous_batching(model, tokenizer, device)

    results = {"static_batching": static, "continuous_batching": continuous}
    path = save_results("bench_batching", results)
    logger.info("Results saved to %s", path)

    W = 16
    print()
    print("=" * 60)
    print(f"{'Metric':<22} | {'Static':>{W}} | {'Continuous':>{W}}")
    print("-" * 60)
    print(f"{'Tokens/sec':<22} | {static['tokens_per_second']:>{W}.2f} | {continuous['tokens_per_second']:>{W}.2f}")
    print(f"{'Mean latency ms':<22} | {static['mean_latency_ms']:>{W}.0f} | {continuous['mean_latency_ms']:>{W}.0f}")
    print(f"{'Max latency ms':<22} | {static['max_latency_ms']:>{W}.0f} | {continuous['max_latency_ms']:>{W}.0f}")
    print(f"{'Total tokens':<22} | {static['total_tokens']:>{W}d} | {continuous['total_tokens']:>{W}d}")
    print("=" * 60)
    print()
    print("Note: static batching pads all sequences to max_new_tokens=",
          STATIC_MAX_NEW_TOKENS, "; continuous batching respects per-request limits.")
    print("Mean latency delta:",
          f"{continuous['mean_latency_ms'] - static['mean_latency_ms']:+.0f} ms "
          f"({'lower is better for continuous' if continuous['mean_latency_ms'] < static['mean_latency_ms'] else 'higher — benefit appears at larger scale/GPU'})")


if __name__ == "__main__":
    run_benchmark()
