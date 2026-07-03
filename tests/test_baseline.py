"""
test_baseline.py — Sanity checks on the HuggingFace baseline for Inferno.

Verifies that the baseline module loads a model, runs generation, and returns
output of the expected shape and token count. All tests run on CPU; no GPU required.

We load the model once at module level (session-scoped via a module-level fixture
pattern) to avoid downloading/loading Qwen2.5-0.5B for every test function.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from inferno.baseline import BaselineResult, load_model, run_baseline
from inferno.utils import RESULTS_DIR

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PROMPTS = ["Hello, my name is", "The capital of France is"]
MAX_NEW_TOKENS = 8   # small so CPU tests are fast
DEVICE = torch.device("cpu")

# Module-level cache so we load the model once across all tests in this file.
_model_cache: dict = {}


@pytest.fixture(scope="module")
def loaded_model():
    """
    Load Qwen2.5-0.5B-Instruct once for all tests in this module.

    Scoped to module so the 500M-parameter model is not re-loaded per test.
    """
    if not _model_cache:
        model, tokenizer, device = load_model(device=DEVICE)
        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer
        _model_cache["device"] = device
    return _model_cache["model"], _model_cache["tokenizer"], _model_cache["device"]


@pytest.fixture(scope="module")
def baseline_result(loaded_model) -> BaselineResult:
    """
    Run baseline once and reuse the result across tests that only need to inspect it.
    """
    model, tokenizer, device = loaded_model
    return run_baseline(
        prompts=PROMPTS,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=1,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaselineOutputs:
    def test_generated_texts_are_non_empty_strings(self, baseline_result: BaselineResult):
        """
        Checks that every prompt yields a non-empty decoded string.

        If generate() failed silently or tokenizer decoded nothing we would get
        empty strings, which would make throughput numbers meaningless.
        """
        assert len(baseline_result.generated_texts) == len(PROMPTS)
        for text in baseline_result.generated_texts:
            assert isinstance(text, str)
            assert len(text.strip()) > 0, f"Empty generated text: {text!r}"

    def test_number_of_outputs_matches_number_of_prompts(self, baseline_result: BaselineResult):
        """Checks that one output text is returned per input prompt."""
        assert baseline_result.num_prompts == len(PROMPTS)
        assert len(baseline_result.generated_texts) == len(PROMPTS)


class TestBaselineMetrics:
    def test_all_numeric_fields_are_positive(self, baseline_result: BaselineResult):
        """
        Checks that every measured numeric field is strictly positive.

        Zero or negative values would indicate a timing or memory measurement bug.
        """
        assert baseline_result.ttft_seconds > 0.0
        assert baseline_result.total_time_seconds > 0.0
        assert baseline_result.tokens_per_second > 0.0
        assert baseline_result.peak_memory_mb > 0.0

    def test_peak_memory_reflects_model_weight_size(self, baseline_result: BaselineResult):
        """
        Checks that peak_memory_mb is > 100 MB after psutil-based tracking.

        Qwen2.5-0.5B has ~500M parameters. In fp32 (4 bytes each) that is ~2 GB
        of model weights alone. Previously tracemalloc returned ~0.2 MB because it
        only saw Python-heap allocations, not C++ tensor memory. With psutil RSS
        we expect hundreds of MB minimum even after Python overhead is stripped.
        """
        assert baseline_result.peak_memory_mb > 100.0, (
            f"peak_memory_mb = {baseline_result.peak_memory_mb:.1f} MB — "
            "expected > 100 MB; psutil RSS may not be capturing tensor memory"
        )

    def test_memory_measurement_is_consistent_across_calls(self, loaded_model):
        """
        Checks that two back-to-back baseline runs report memory within 10% of each other.

        If memory were non-deterministic or the tracker were measuring the wrong
        thing, the readings would vary wildly between calls with the same model loaded.
        """
        model, tokenizer, device = loaded_model
        kwargs = dict(prompts=["Hi"], max_new_tokens=4, batch_size=1,
                      model=model, tokenizer=tokenizer, device=device)
        r1 = run_baseline(**kwargs)
        r2 = run_baseline(**kwargs)
        # Allow 10% relative difference — RSS includes OS page-rounding noise
        ratio = max(r1.peak_memory_mb, r2.peak_memory_mb) / max(min(r1.peak_memory_mb, r2.peak_memory_mb), 1.0)
        assert ratio < 1.10, (
            f"Memory readings differ by more than 10%: {r1.peak_memory_mb:.1f} vs {r2.peak_memory_mb:.1f} MB"
        )

    def test_ttft_is_less_than_total_time(self, baseline_result: BaselineResult):
        """
        Checks that time-to-first-token is always a subset of total generation time.

        TTFT measures a single forward pass; total time covers all batches and all
        decode steps, so TTFT must always be smaller.
        """
        assert baseline_result.ttft_seconds < baseline_result.total_time_seconds

    def test_tokens_per_second_is_consistent_with_timing(self, baseline_result: BaselineResult):
        """
        Checks that tokens/sec is arithmetically consistent with total time and
        the maximum possible token count (num_prompts * max_new_tokens).

        We allow the actual count to be less because EOS tokens may be generated
        before max_new_tokens is reached.
        """
        max_possible_tokens = len(PROMPTS) * MAX_NEW_TOKENS
        implied_tokens = baseline_result.tokens_per_second * baseline_result.total_time_seconds
        assert implied_tokens <= max_possible_tokens * 1.05  # 5% tolerance for float rounding


class TestBaselineResultSaving:
    def test_json_file_is_created_in_results_dir(self, loaded_model):
        """
        Checks that run_baseline() always writes a JSON file to results/.

        We run a fresh call so we can check the newest file created after this call.
        """
        before_files = set(RESULTS_DIR.glob("baseline_*.json"))

        model, tokenizer, device = loaded_model
        run_baseline(
            prompts=["Test prompt"],
            max_new_tokens=4,
            batch_size=1,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        after_files = set(RESULTS_DIR.glob("baseline_*.json"))
        new_files = after_files - before_files
        assert len(new_files) == 1, f"Expected 1 new result file, got {len(new_files)}"

    def test_json_file_is_valid_and_contains_required_keys(self, loaded_model):
        """
        Checks that the saved JSON is parseable and contains all BaselineResult fields.
        """
        model, tokenizer, device = loaded_model
        run_baseline(
            prompts=["Check JSON keys"],
            max_new_tokens=4,
            batch_size=1,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        # Find the most recent baseline file
        files = sorted(RESULTS_DIR.glob("baseline_*.json"))
        assert files, "No baseline result files found"
        latest = files[-1]

        with open(latest) as f:
            data = json.load(f)

        required_keys = {
            "model_id", "num_prompts", "max_new_tokens", "batch_size",
            "ttft_seconds", "total_time_seconds", "tokens_per_second",
            "peak_memory_mb", "generated_texts",
        }
        assert required_keys.issubset(data.keys()), f"Missing keys: {required_keys - data.keys()}"
