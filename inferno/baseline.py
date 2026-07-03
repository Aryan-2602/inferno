"""
baseline.py — HuggingFace naive generate() baseline for Inferno.

Loads a model and runs standard HuggingFace generate() with no optimizations.
This is the reference implementation that all optimizations are measured against.
Throughput (tokens/sec) and peak memory (MB) are recorded for every run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferno.utils import GpuMemoryTracker, get_logger, save_results, wall_time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_BATCH_SIZE = 1

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """All measurements from a single baseline generate() run."""

    model_id: str
    num_prompts: int
    max_new_tokens: int
    batch_size: int
    ttft_seconds: float          # time to first token for the first batch
    total_time_seconds: float    # wall time covering all batches
    tokens_per_second: float     # total new tokens / total wall time
    peak_memory_mb: float        # peak memory during the full run
    generated_texts: list[str]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[torch.device] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Load model and tokenizer from HuggingFace Hub onto the resolved device.

    Uses float32 on CPU to keep memory arithmetic predictable; on CUDA we use
    the model's native dtype (bfloat16 for Qwen2.5).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading %s on %s", model_id, device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    logger.info("Model loaded — parameters: %dM", sum(p.numel() for p in model.parameters()) // 1_000_000)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def run_baseline(
    prompts: list[str],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_id: str = DEFAULT_MODEL_ID,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[torch.device] = None,
) -> BaselineResult:
    """
    Run HuggingFace generate() on prompts and return measured performance.

    Processes prompts in batches of batch_size. TTFT is measured on the first
    batch only (single-token decode step after prefill). Total throughput counts
    every generated token across all batches.

    If model/tokenizer are passed in they are reused (useful for tests);
    otherwise they are loaded from model_id.
    """
    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_model(model_id, device)

    generated_texts: list[str] = []
    total_new_tokens = 0
    ttft_seconds = 0.0

    mem_tracker = GpuMemoryTracker(device)
    run_start = wall_time()

    with mem_tracker:
        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx : batch_idx + batch_size]

            # BASELINE: padding to longest in batch — no dynamic batching or
            # memory-efficient attention; this is the naive control path.
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            input_len = inputs["input_ids"].shape[1]

            # ---- Measure TTFT on first batch only ----
            if batch_idx == 0:
                # BASELINE: single forward pass to get the first new token; we
                # re-run the full generate() below so prefill cost is counted
                # twice for the first batch — acceptable for a control baseline.
                ttft_start = wall_time()
                with torch.no_grad():
                    _ = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                ttft_seconds = wall_time() - ttft_start

            # ---- Full generation ----
            with torch.no_grad():
                # BASELINE: using greedy decoding (do_sample=False); no
                # speculative decoding, no KV cache compression, no custom
                # attention kernel.
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the newly generated portion
            new_ids = output_ids[:, input_len:]
            total_new_tokens += new_ids.numel()

            decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            generated_texts.extend(decoded)

    total_time_seconds = wall_time() - run_start
    tokens_per_second = total_new_tokens / total_time_seconds if total_time_seconds > 0 else 0.0

    result = BaselineResult(
        model_id=model_id,
        num_prompts=len(prompts),
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        ttft_seconds=ttft_seconds,
        total_time_seconds=total_time_seconds,
        tokens_per_second=tokens_per_second,
        peak_memory_mb=mem_tracker.peak_mb,
        generated_texts=generated_texts,
    )

    path = save_results("baseline", asdict(result))
    logger.info("Results saved to %s", path)
    logger.info(
        "Throughput: %.2f tok/s | Peak memory: %.1f MB | TTFT: %.3f s",
        tokens_per_second,
        mem_tracker.peak_mb,
        ttft_seconds,
    )
    return result
