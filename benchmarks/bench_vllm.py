"""
bench_vllm.py — vLLM vs Inferno custom implementation comparison.

Checks for vLLM at import time; exits cleanly with install instructions if
not available. When vLLM is present, benchmarks the same model and prompts
used in bench_gpu.py and prints a head-to-head comparison table against the
most recent Inferno bench_gpu_*.json results.

Requires GPU — vLLM will fail at LLM() construction without CUDA.

Install vLLM:
    pip install -r requirements_vllm.txt   (single-package file: vllm)

Run:
    source .venv/bin/activate
    python benchmarks/bench_vllm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- vLLM availability gate (graceful exit, not crash) ---
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM is not installed.")
    print()
    print("To install:")
    print("    pip install vllm")
    print("    # or: pip install -r requirements_vllm.txt")
    print()
    print("Note: vLLM requires a CUDA-capable GPU (Pascal or newer).")
    sys.exit(0)

import json

import numpy as np
import torch

from inferno.baseline import DEFAULT_MODEL_ID
from inferno.utils import RESULTS_DIR, get_logger, save_results, wall_time

# ---------------------------------------------------------------------------
# Constants — identical to bench_gpu.py so comparison is apples-to-apples
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

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_latest_inferno_gpu_results() -> dict | None:
    """
    Load the most recent bench_gpu_*.json from results/.

    Returns None if no file exists — the comparison table degrades gracefully
    to a vLLM-only display.
    """
    files = sorted(RESULTS_DIR.glob("bench_gpu_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_vllm_benchmark() -> dict:
    """
    Load the same model used in baseline.py via vLLM and measure throughput + latency.

    Per-request latency: each prompt timed individually (sequential calls).
    Throughput: all prompts submitted at once so vLLM can batch internally.
    """
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. bench_vllm.py requires a GPU.")
        sys.exit(1)

    logger.info("Loading %s via vLLM ...", DEFAULT_MODEL_ID)
    # dtype="bfloat16" matches the GPU dtype used in baseline.py
    llm = LLM(model=DEFAULT_MODEL_ID, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    # Per-request latency: one request at a time
    logger.info("Measuring per-request latency (%d prompts, sequential) ...", len(PROMPTS))
    latencies_ms: list[float] = []
    for prompt in PROMPTS:
        t0 = wall_time()
        _ = llm.generate([prompt], sampling_params)
        latencies_ms.append((wall_time() - t0) * 1000.0)

    # Throughput: all prompts at once — vLLM continuous batching fires internally
    logger.info("Measuring throughput (all %d prompts batched) ...", len(PROMPTS))
    t0 = wall_time()
    outputs = llm.generate(PROMPTS, sampling_params)
    total_time = wall_time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / total_time if total_time > 0 else 0.0

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = sum(latencies_ms) / len(latencies_ms)

    logger.info(
        "vLLM: %.1f tok/s | p50=%.0f ms | p95=%.0f ms | p99=%.0f ms",
        tps, p50, p95, p99,
    )
    return {
        "model": DEFAULT_MODEL_ID,
        "tokens_per_second": tps,
        "total_tokens": total_tokens,
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "max_latency_ms": max(latencies_ms),
    }


def _winner(val_a: float, val_b: float, higher_better: bool) -> tuple[str, str]:
    """Return (winner_label, sign_str) for a metric, with a simple Why explanation."""
    a_wins = (val_a > val_b) if higher_better else (val_a < val_b)
    if a_wins:
        return "vLLM", "↑" if higher_better else "↓"
    return "Inferno", "↓" if higher_better else "↑"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    """Run vLLM benchmark and compare against stored Inferno GPU results."""
    vllm_results = run_vllm_benchmark()

    all_results = {"vllm": vllm_results}
    path = save_results("bench_vllm", all_results)
    logger.info("Results saved to %s", path)

    inferno = load_latest_inferno_gpu_results()

    print()
    print("=" * 92)
    print("  vLLM vs Inferno — GPU Comparison")
    print(f"  (Inferno source: {sorted(RESULTS_DIR.glob('bench_gpu_*.json'))[-1].name if inferno else 'not found'})")
    print("=" * 92)

    W = 14
    L = 26  # why-column approx

    if inferno:
        ib = inferno.get("cache_benchmark", {}).get("baseline", {})
        i_tps = ib.get("tokens_per_second", float("nan"))

        # Inferno static batching latencies (present in results produced after Task 1)
        isb = inferno.get("batching_benchmark", {}).get("static_batching", {})
        i_p50 = isb.get("p50_latency_ms", float("nan"))
        i_p95 = isb.get("p95_latency_ms", float("nan"))
        i_p99 = isb.get("p99_latency_ms", float("nan"))

        v_tps = vllm_results["tokens_per_second"]
        v_p50 = vllm_results["p50_latency_ms"]
        v_p95 = vllm_results["p95_latency_ms"]
        v_p99 = vllm_results["p99_latency_ms"]

        header = f"{'Metric':<22} | {'vLLM':>{W}} | {'Inferno':>{W}} | {'Winner':<10} | Why"
        print(header)
        print("-" * 92)

        # Tokens/sec: higher is better
        w_tps, _ = _winner(v_tps, i_tps, higher_better=True)
        why_tps = (
            "PagedAttention + fused kernels batch efficiently"
            if w_tps == "vLLM"
            else "Lower overhead at this concurrency level"
        )
        print(f"{'Tokens/sec':<22} | {v_tps:>{W}.1f} | {i_tps:>{W}.1f} | {w_tps:<10} | {why_tps}")

        # p50: lower is better
        if not (i_p50 != i_p50):  # nan check
            w_p50, _ = _winner(v_p50, i_p50, higher_better=False)
            why_p50 = (
                "Fused CUDA kernels, no Python dispatch per token"
                if w_p50 == "vLLM"
                else "Inferno measured single-sequence; vLLM measured individually too"
            )
            print(f"{'p50 latency ms':<22} | {v_p50:>{W}.0f} | {i_p50:>{W}.0f} | {w_p50:<10} | {why_p50}")

            w_p95, _ = _winner(v_p95, i_p95, higher_better=False)
            why_p95 = (
                "vLLM preempts long tail via scheduling"
                if w_p95 == "vLLM"
                else "Static batching keeps sequences together"
            )
            print(f"{'p95 latency ms':<22} | {v_p95:>{W}.0f} | {i_p95:>{W}.0f} | {w_p95:<10} | {why_p95}")

            w_p99, _ = _winner(v_p99, i_p99, higher_better=False)
            why_p99 = (
                "vLLM continuous batching drains tail requests faster"
                if w_p99 == "vLLM"
                else "Low concurrency; no advantage for continuous batching"
            )
            print(f"{'p99 latency ms':<22} | {v_p99:>{W}.0f} | {i_p99:>{W}.0f} | {w_p99:<10} | {why_p99}")
        else:
            print(f"{'p50 latency ms':<22} | {v_p50:>{W}.0f} | {'N/A (re-run bench_gpu)':>{W}} | {'—':<10} |")
            print(f"{'p95 latency ms':<22} | {v_p95:>{W}.0f} | {'N/A':>{W}} | {'—':<10} |")
            print(f"{'p99 latency ms':<22} | {v_p99:>{W}.0f} | {'N/A':>{W}} | {'—':<10} |")

    else:
        print(f"{'Metric':<22} | {'vLLM':>{W}}")
        print("-" * 42)
        print(f"{'Tokens/sec':<22} | {vllm_results['tokens_per_second']:>{W}.1f}")
        print(f"{'p50 latency ms':<22} | {vllm_results['p50_latency_ms']:>{W}.0f}")
        print(f"{'p95 latency ms':<22} | {vllm_results['p95_latency_ms']:>{W}.0f}")
        print(f"{'p99 latency ms':<22} | {vllm_results['p99_latency_ms']:>{W}.0f}")
        print()
        print("(No Inferno bench_gpu results found. Run benchmarks/bench_gpu.py first.)")

    print("=" * 92)
    print()
    print("Note: Inferno latency is from static batching (4 requests/batch).")
    print("      vLLM latency is per-request (individual calls, sequential).")
    print("      Throughput comparison is most meaningful: vLLM batches internally.")


if __name__ == "__main__":
    run_benchmark()
