"""
bench_batching.py — Continuous batching throughput benchmark for Inferno.

Compares two scheduling strategies on the same workload:

  Static batching   — group prompts into padded batches; call model.generate()
                      with max_new_tokens = max across the batch. All sequences
                      in a batch finish at the same time.

  Continuous batching — each request carries its own max_new_tokens budget;
                        short sequences finish early and free their slot without
                        stalling longer-running ones.

# INTEGRITY: Both strategies must process identical workloads (same prompts,
# same per-request token budgets) so their throughput numbers are comparable.
# Phase 2 violated this: static used STATIC_MAX_NEW_TOKENS=32 for all 8 requests
# (256 tokens), continuous used per-request budgets (120 tokens). The 2x token
# difference made the throughput numbers incomparable. Fixed here by:
#   1. Building a single canonical request list with build_workload().
#   2. Static batching slices output to each request's individual budget.
#   3. Asserting total_tokens_static == total_tokens_continuous at the end.

Saves results to results/ as JSON and prints a comparison table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from inferno.baseline import load_model
from inferno.engine import ContinuousBatchingEngine, Request, SchedulerConfig
from inferno.utils import get_logger, save_results, wall_time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Heterogeneous workload spec: (prompt, max_new_tokens) pairs.
# Mixed token budgets make the scheduling difference visible — short sequences
# (budget=8) would wait for long ones (budget=32) in static batching but not
# in continuous batching.
WORKLOAD_SPEC: list[tuple[str, int]] = [
    ("Hi", 8),
    ("The capital of France is", 8),
    ("Explain gravity briefly", 16),
    ("What is 2+2?", 8),
    ("Tell me about machine learning in detail", 32),
    ("Name three planets", 8),
    ("Summarise quantum mechanics", 32),
    ("What color is the sky?", 8),
]

STATIC_BATCH_SIZE = 4   # sequences per model.generate() call in static mode

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical workload builder
# ---------------------------------------------------------------------------

def build_workload() -> list[Request]:
    """
    Build the canonical fixed request list used by both benchmarks.

    Both static and continuous batching call this function so they operate on
    structurally identical workloads. Calling it twice produces independent
    Request objects (different request_ids/arrival_times) with identical
    prompt/budget pairs.
    """
    return [Request(prompt=p, max_new_tokens=tok) for p, tok in WORKLOAD_SPEC]


# ---------------------------------------------------------------------------
# Static batching
# ---------------------------------------------------------------------------

def run_static_batching(requests: list[Request], model, tokenizer, device: torch.device) -> dict:
    """
    Naïve static batching: group requests into padded batches and call
    model.generate() once per batch.

    Token budget: each batch uses max(req.max_new_tokens for req in batch) as
    the generate budget so no sequence is cut short. The output is then sliced
    to each request's individual budget before counting tokens — this is the
    parity fix that ensures total_tokens matches continuous batching.

    Latency: every sequence in a batch finishes at the same wall time (the batch
    duration), even if its individual budget was shorter. This inflates mean
    latency for short-budget sequences relative to continuous batching.
    """
    eos_id: int = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    total_tokens = 0
    latencies_ms: list[float] = []
    start_all = wall_time()

    for batch_start in range(0, len(requests), STATIC_BATCH_SIZE):
        batch = requests[batch_start : batch_start + STATIC_BATCH_SIZE]
        batch_prompts = [req.prompt for req in batch]

        # Use max budget in the batch so no sequence is truncated.
        # Shorter sequences' outputs are sliced below.
        batch_max_tok = max(req.max_new_tokens for req in batch)

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
                max_new_tokens=batch_max_tok,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed_ms = (wall_time() - t0) * 1000.0

        for i, req in enumerate(batch):
            # INTEGRITY: slice to req.max_new_tokens so we count only the tokens
            # this request was budgeted for — same as continuous batching.
            gen_slice = out[i, input_len : input_len + req.max_new_tokens]

            # Mirror engine EOS logic: stop counting at the first EOS token
            # (include it) so static and continuous agree on early-stop cases.
            eos_pos = (gen_slice == eos_id).nonzero(as_tuple=True)[0]
            count = int(eos_pos[0].item()) + 1 if len(eos_pos) > 0 else int(gen_slice.shape[0])
            total_tokens += count

            # Every sequence in the batch waits the full batch duration.
            latencies_ms.append(elapsed_ms)

    total_time = wall_time() - start_all
    tps = total_tokens / total_time if total_time > 0 else 0.0
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    max_lat = max(latencies_ms)
    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))

    logger.info(
        "Static batching: %.2f tok/s | mean lat=%.0f ms | max lat=%.0f ms | tokens=%d",
        tps, mean_lat, max_lat, total_tokens,
    )
    return {
        "tokens_per_second": tps,
        "mean_latency_ms": mean_lat,
        "max_latency_ms": max_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Continuous batching
# ---------------------------------------------------------------------------

def run_continuous_batching(requests: list[Request], model, tokenizer, device: torch.device) -> dict:
    """
    Continuous batching via ContinuousBatchingEngine.

    Each request carries its own max_new_tokens budget. Short sequences exit
    early, freeing their slot for the next waiting request without stalling
    longer-running ones. Mean latency for short sequences is therefore lower
    than in static batching.
    """
    config = SchedulerConfig(max_batch_size=STATIC_BATCH_SIZE, max_prefill_chunk_size=512)
    engine = ContinuousBatchingEngine(model=model, tokenizer=tokenizer, config=config, device=device)

    for req in requests:
        engine.submit(req)

    start_all = wall_time()
    completed = engine.run_until_complete()
    total_time = wall_time() - start_all

    latencies_ms = [cr.latency_ms for cr in completed]
    total_tokens = sum(cr.tokens_generated for cr in completed)
    tps = total_tokens / total_time if total_time > 0 else 0.0
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    max_lat = max(latencies_ms)
    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))

    logger.info(
        "Continuous batching: %.2f tok/s | mean lat=%.0f ms | max lat=%.0f ms | tokens=%d",
        tps, mean_lat, max_lat, total_tokens,
    )
    return {
        "tokens_per_second": tps,
        "mean_latency_ms": mean_lat,
        "max_latency_ms": max_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    """
    Run static and continuous batching on the same workload and print a
    comparison table. Asserts workload parity before saving results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running batching benchmark on device: %s", device)
    model, tokenizer, device = load_model(device=device)

    # Build two independent but structurally identical request lists.
    static_requests = build_workload()
    continuous_requests = build_workload()

    logger.info("=== Static batching ===")
    static = run_static_batching(static_requests, model, tokenizer, device)

    logger.info("=== Continuous batching ===")
    continuous = run_continuous_batching(continuous_requests, model, tokenizer, device)

    # INTEGRITY: both strategies must have processed the same total token count.
    # A mismatch means the workload definitions diverged (e.g. EOS triggered at
    # different positions), which would make throughput numbers incomparable.
    if static["total_tokens"] != continuous["total_tokens"]:
        raise ValueError(
            f"Workload parity violation: static produced {static['total_tokens']} tokens "
            f"but continuous produced {continuous['total_tokens']} tokens. "
            "Throughput numbers are not comparable — investigate EOS differences."
        )
    logger.info(
        "Workload parity check passed: both strategies produced %d tokens.",
        static["total_tokens"],
    )

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
    print(f"{'p50 latency ms':<22} | {static['p50_latency_ms']:>{W}.0f} | {continuous['p50_latency_ms']:>{W}.0f}")
    print(f"{'p95 latency ms':<22} | {static['p95_latency_ms']:>{W}.0f} | {continuous['p95_latency_ms']:>{W}.0f}")
    print(f"{'p99 latency ms':<22} | {static['p99_latency_ms']:>{W}.0f} | {continuous['p99_latency_ms']:>{W}.0f}")
    print(f"{'Max latency ms':<22} | {static['max_latency_ms']:>{W}.0f} | {continuous['max_latency_ms']:>{W}.0f}")
    print(f"{'Total tokens':<22} | {static['total_tokens']:>{W}d} | {continuous['total_tokens']:>{W}d}")
    print("=" * 60)
    mean_delta = continuous["mean_latency_ms"] - static["mean_latency_ms"]
    print(f"\nMean latency delta: {mean_delta:+.0f} ms "
          f"({'continuous wins — short sequences finish early' if mean_delta < 0 else 'static wins on CPU — batched matmuls are cheaper than sequential passes'})")


if __name__ == "__main__":
    run_benchmark()
