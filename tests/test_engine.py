"""
test_engine.py — Correctness tests for the continuous batching scheduler.

Verifies scheduling invariants — not generation quality. Every test runs on CPU
with max_new_tokens=4 to keep wall time under a few seconds per test.

Correctness properties verified:
  - A single request is completed with non-empty output and positive latency.
  - Two simultaneously-submitted requests both complete.
  - max_batch_size is never exceeded (scheduler respects the slot limit).
  - latency_ms is a positive float measuring real elapsed time.
  - A request submitted after the engine has started running still completes.
  - Identical prompts produce identical output (greedy decode is deterministic).
"""

from __future__ import annotations

import time

import pytest
import torch

from inferno.baseline import load_model
from inferno.engine import (
    ContinuousBatchingEngine,
    CompletedRequest,
    Request,
    SchedulerConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 4     # fast on CPU
PROMPT_A = "Hello"
PROMPT_B = "The sky is"
PROMPT_C = "Once upon a time"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_model_cache: dict = {}


@pytest.fixture(scope="module")
def loaded_model():
    """
    Load Qwen2.5-0.5B-Instruct once for the whole test module to avoid
    the multi-second load time on every test function.
    """
    if not _model_cache:
        model, tokenizer, device = load_model(device=torch.device("cpu"))
        _model_cache.update(model=model, tokenizer=tokenizer, device=device)
    return _model_cache["model"], _model_cache["tokenizer"], _model_cache["device"]


@pytest.fixture
def engine(loaded_model):
    """
    Return a fresh ContinuousBatchingEngine instance for each test so that
    scheduler state from one test cannot bleed into another.
    """
    model, tokenizer, device = loaded_model
    config = SchedulerConfig(max_batch_size=2, max_prefill_chunk_size=128)
    return ContinuousBatchingEngine(model=model, tokenizer=tokenizer, config=config, device=device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleRequest:
    def test_single_request_completes(self, engine: ContinuousBatchingEngine):
        """
        A single submitted request must appear in the completed list returned by
        run_until_complete(). If the scheduler never evicts the request the list
        would be empty and this test would catch that.
        """
        req = Request(prompt=PROMPT_A, max_new_tokens=MAX_NEW_TOKENS)
        engine.submit(req)

        results = engine.run_until_complete()

        assert len(results) == 1
        cr = results[0]
        assert cr.request_id == req.request_id

    def test_single_request_output_non_empty(self, engine: ContinuousBatchingEngine):
        """
        The decoded output for a completed request must be a non-empty string.
        An empty output would indicate that generated_ids were empty or that the
        tokenizer decoded nothing (EOS-only output is stripped via skip_special_tokens).
        """
        engine.submit(Request(prompt=PROMPT_A, max_new_tokens=MAX_NEW_TOKENS))
        results = engine.run_until_complete()

        assert isinstance(results[0].output_text, str)
        assert len(results[0].output_text) > 0

    def test_single_request_tokens_generated_matches_limit(self, engine: ContinuousBatchingEngine):
        """
        tokens_generated must be at most max_new_tokens (EOS may stop it early)
        and at least 1 (the prefill step always produces one token).
        """
        engine.submit(Request(prompt=PROMPT_A, max_new_tokens=MAX_NEW_TOKENS))
        results = engine.run_until_complete()

        assert 1 <= results[0].tokens_generated <= MAX_NEW_TOKENS


class TestMultipleRequests:
    def test_two_requests_both_complete(self, engine: ContinuousBatchingEngine):
        """
        Two simultaneously-submitted requests must both appear in the output list.
        Checks that the scheduler drains both sequences from the running queue.
        """
        req_a = Request(prompt=PROMPT_A, max_new_tokens=MAX_NEW_TOKENS)
        req_b = Request(prompt=PROMPT_B, max_new_tokens=MAX_NEW_TOKENS)
        engine.submit(req_a)
        engine.submit(req_b)

        results = engine.run_until_complete()
        assert len(results) == 2

        ids = {cr.request_id for cr in results}
        assert req_a.request_id in ids
        assert req_b.request_id in ids

    def test_max_batch_size_never_exceeded(self, loaded_model):
        """
        At every step(), the number of running sequences must not exceed
        max_batch_size. We instrument the engine to assert this invariant
        inside each step() call by wrapping _admit_waiting.
        """
        model, tokenizer, device = loaded_model
        max_bs = 2
        config = SchedulerConfig(max_batch_size=max_bs, max_prefill_chunk_size=128)
        eng = ContinuousBatchingEngine(model=model, tokenizer=tokenizer, config=config, device=device)

        for i in range(4):   # submit more than max_batch_size
            eng.submit(Request(prompt=PROMPT_A, max_new_tokens=MAX_NEW_TOKENS, request_id=str(i)))

        violations: list[int] = []
        original_step = eng.step

        def tracked_step():
            completed = original_step()
            if len(eng._running) > max_bs:
                violations.append(len(eng._running))
            return completed

        eng.step = tracked_step  # type: ignore[method-assign]
        while eng._waiting or eng._running:
            eng.step()

        assert violations == [], f"Running queue exceeded max_batch_size on steps: {violations}"


class TestLatency:
    def test_latency_ms_is_positive(self, engine: ContinuousBatchingEngine):
        """
        latency_ms must be a positive float. Zero or negative would mean
        wall_time() measurement is broken or start_time was set incorrectly.
        """
        engine.submit(Request(prompt=PROMPT_A, max_new_tokens=MAX_NEW_TOKENS))
        results = engine.run_until_complete()

        assert results[0].latency_ms > 0.0

    def test_latency_ms_is_plausible(self, engine: ContinuousBatchingEngine):
        """
        latency_ms must be less than 120 seconds. Catches the case where start_time
        was accidentally set to 0 or epoch (very large latency) rather than wall time.
        """
        engine.submit(Request(prompt=PROMPT_A, max_new_tokens=MAX_NEW_TOKENS))
        results = engine.run_until_complete()

        assert results[0].latency_ms < 120_000.0, (
            f"latency_ms={results[0].latency_ms:.1f} is implausibly large"
        )


class TestLateSubmission:
    def test_late_submitted_request_completes(self, engine: ContinuousBatchingEngine):
        """
        A request submitted after the engine has started processing other requests
        must still be completed by run_until_complete().

        This checks the scheduler's waiting_queue admits new requests mid-flight
        and does not starve them out.
        """
        req_early = Request(prompt=PROMPT_A, max_new_tokens=MAX_NEW_TOKENS)
        req_late = Request(prompt=PROMPT_C, max_new_tokens=MAX_NEW_TOKENS)

        engine.submit(req_early)

        # Run one step so req_early is in-flight, then submit req_late
        engine.step()
        engine.submit(req_late)

        # Drain remaining
        remaining = engine.run_until_complete()

        all_ids = {cr.request_id for cr in remaining}
        assert req_late.request_id in all_ids, "Late-submitted request was never completed"


class TestDeterminism:
    def test_identical_prompts_produce_identical_output(self, engine: ContinuousBatchingEngine):
        """
        Two separate requests with the same prompt must produce exactly the same
        output_text. Greedy decoding is deterministic, so any difference would
        indicate accidental state sharing between sequences (e.g., reused caches).
        """
        req_1 = Request(prompt=PROMPT_B, max_new_tokens=MAX_NEW_TOKENS)
        req_2 = Request(prompt=PROMPT_B, max_new_tokens=MAX_NEW_TOKENS)

        engine.submit(req_1)
        results_1 = engine.run_until_complete()

        # Fresh engine for second run to avoid any residual state
        model, tokenizer, device = (
            engine.model, engine.tokenizer, engine.device
        )
        config = engine.config
        eng2 = ContinuousBatchingEngine(model=model, tokenizer=tokenizer, config=config, device=device)
        eng2.submit(req_2)
        results_2 = eng2.run_until_complete()

        assert results_1[0].output_text == results_2[0].output_text, (
            f"Non-deterministic output:\n  run1: {results_1[0].output_text!r}\n  run2: {results_2[0].output_text!r}"
        )
