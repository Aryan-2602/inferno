"""
bench_speculative.py — Speculative decoding vs autoregressive baseline benchmark.

Compares two generation strategies on the same prompts:
  Autoregressive — target model (Qwen2.5-1.5B) generates one token at a time.
  Speculative    — draft model (Qwen2.5-0.5B) proposes gamma tokens per step;
                   target model verifies in one forward pass.

Both strategies use greedy / near-greedy decoding so outputs are comparable.

Measurements:
  tokens/sec, acceptance_rate (speculative only), mean latency ms per prompt.

Results saved to results/bench_speculative_{timestamp}.json.

Requires GPU — exits with error if CUDA is not available.
Run scripts/check_gpu.py first to validate the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

if not torch.cuda.is_available():
    print("ERROR: No CUDA device found. bench_speculative.py requires a GPU.")
    print("Run scripts/check_gpu.py first to diagnose the environment.")
    sys.exit(1)

import time

from inferno.speculative import SpecdecEngine, load_draft_and_target
from inferno.utils import get_logger, save_results, wall_time

# ---------------------------------------------------------------------------
# Constants — same prompt list as bench_gpu.py for cross-script comparability
# ---------------------------------------------------------------------------

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

MAX_NEW_TOKENS = 64
GAMMA = 4   # draft tokens per speculative step

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Autoregressive baseline on the target model only
# ---------------------------------------------------------------------------

def run_autoregressive_baseline(target_model, target_tokenizer, device: torch.device) -> dict:
    """
    Run autoregressive generation on the target model as the comparison baseline.

    One token at a time, no draft model. Measures tokens/sec and per-prompt latency.
    """
    latencies_ms: list[float] = []
    total_tokens = 0
    t_all = wall_time()

    for prompt in PROMPTS:
        enc = target_tokenizer(prompt, return_tensors="pt").to(device)
        input_len = enc["input_ids"].shape[1]

        t0 = wall_time()
        with torch.no_grad():
            out = target_model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=target_tokenizer.pad_token_id,
            )
        elapsed_ms = (wall_time() - t0) * 1000.0

        new_tokens = out.shape[1] - input_len
        total_tokens += new_tokens
        latencies_ms.append(elapsed_ms)

    total_time = wall_time() - t_all
    tps = total_tokens / total_time if total_time > 0 else 0.0
    mean_lat = sum(latencies_ms) / len(latencies_ms)

    logger.info(
        "Autoregressive: %.1f tok/s | mean lat=%.0f ms | tokens=%d",
        tps, mean_lat, total_tokens,
    )
    return {
        "strategy": "autoregressive",
        "model": target_model.config._name_or_path if hasattr(target_model.config, "_name_or_path") else "target",
        "tokens_per_second": tps,
        "mean_latency_ms": mean_lat,
        "total_tokens": total_tokens,
        "acceptance_rate": None,
    }


# ---------------------------------------------------------------------------
# Speculative decoding run
# ---------------------------------------------------------------------------

def run_speculative(engine: SpecdecEngine) -> dict:
    """
    Run speculative decoding on all PROMPTS, collect throughput and acceptance rate.
    """
    latencies_ms: list[float] = []
    total_tokens = 0
    acceptance_rates: list[float] = []
    t_all = wall_time()

    for prompt in PROMPTS:
        result = engine.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        latencies_ms.append(result["time_seconds"] * 1000.0)
        total_tokens += result["tokens_generated"]
        acceptance_rates.append(result["acceptance_rate"])

    total_time = wall_time() - t_all
    tps = total_tokens / total_time if total_time > 0 else 0.0
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    mean_ar = sum(acceptance_rates) / len(acceptance_rates)

    logger.info(
        "Speculative (gamma=%d): %.1f tok/s | mean lat=%.0f ms | acceptance=%.2f | tokens=%d",
        engine.gamma, tps, mean_lat, mean_ar, total_tokens,
    )
    return {
        "strategy": "speculative",
        "gamma": engine.gamma,
        "tokens_per_second": tps,
        "mean_latency_ms": mean_lat,
        "total_tokens": total_tokens,
        "acceptance_rate": mean_ar,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    """Run autoregressive and speculative decoding, print comparison table, save JSON."""
    device = torch.device("cuda")
    logger.info("Speculative decoding benchmark on: %s", torch.cuda.get_device_name(device))

    draft_model, draft_tok, target_model, target_tok, device = load_draft_and_target(device=device)

    logger.info("=== Autoregressive baseline (target model only) ===")
    ar_results = run_autoregressive_baseline(target_model, target_tok, device)

    logger.info("=== Speculative decoding (draft=0.5B, target=1.5B, gamma=%d) ===", GAMMA)
    engine = SpecdecEngine(
        draft_model=draft_model,
        draft_tokenizer=draft_tok,
        target_model=target_model,
        target_tokenizer=target_tok,
        device=device,
        gamma=GAMMA,
    )
    spec_results = run_speculative(engine)

    all_results = {
        "device": torch.cuda.get_device_name(device),
        "gamma": GAMMA,
        "autoregressive": ar_results,
        "speculative": spec_results,
    }
    path = save_results("bench_speculative", all_results)
    logger.info("Results saved to %s", path)

    # ---- Print comparison table ----
    W = 18
    ar = ar_results
    sp = spec_results

    print()
    print("=" * 68)
    print("  Speculative Decoding vs Autoregressive (target model only)")
    print("=" * 68)
    print(f"{'Metric':<24} | {'Autoregressive':>{W}} | {'Speculative':>{W}}")
    print("-" * 68)
    print(f"{'Tokens/sec':<24} | {ar['tokens_per_second']:>{W}.1f} | {sp['tokens_per_second']:>{W}.1f}")
    print(f"{'Mean latency ms':<24} | {ar['mean_latency_ms']:>{W}.0f} | {sp['mean_latency_ms']:>{W}.0f}")
    print(f"{'Acceptance rate':<24} | {'—':>{W}} | {sp['acceptance_rate']:>{W}.3f}")
    print(f"{'Total tokens':<24} | {ar['total_tokens']:>{W}d} | {sp['total_tokens']:>{W}d}")
    print("=" * 68)

    speedup = sp["tokens_per_second"] / max(ar["tokens_per_second"], 1e-10)
    print(f"\nSpeedup: {speedup:.2f}x  |  Draft acceptance rate: {sp['acceptance_rate']:.1%}")
    print(
        "Note: acceptance_rate = mean fraction of draft tokens accepted per step. "
        "Higher is better (more draft tokens reused, fewer target calls)."
    )


if __name__ == "__main__":
    run_benchmark()
