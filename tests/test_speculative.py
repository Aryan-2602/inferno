"""
test_speculative.py — Tests for the speculative decoding engine.

Verifies:
  - SpecdecEngine initialises correctly with mocked draft/target models.
  - generate() returns all required output keys.
  - acceptance_rate is always within [0, 1].
  - save_results("bench_speculative", ...) creates a results/ JSON file,
    confirming the save path that bench_speculative.py depends on.

All tests run on CPU using unittest.mock — no real model weights required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import pytest

from inferno.speculative import SpecdecEngine

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 100
PROMPT_INPUT_IDS = [3, 4, 5]   # three fake token ids for the mock prompt


def _make_mock_output(seq_len: int) -> MagicMock:
    """
    Return a mock model output with logits shaped [1, seq_len, VOCAB_SIZE].

    Token 5 always receives high logit (10.0), so softmax concentrates on token 5.
    Token 2 is eos_token_id — never chosen so tests run to max_new_tokens.
    """
    out = MagicMock()
    out.logits = torch.zeros(1, seq_len, VOCAB_SIZE)
    out.logits[0, :, 5] = 10.0
    out.past_key_values = MagicMock()
    return out


def _make_mock_model() -> MagicMock:
    """
    Mock a HuggingFace model whose __call__ returns _make_mock_output.

    The mock reads input_ids.shape[1] to match the real signature:
        model(input_ids=tensor, past_key_values=..., use_cache=True)
    """
    model = MagicMock()
    model.eval.return_value = None
    model.side_effect = lambda **kwargs: _make_mock_output(kwargs["input_ids"].shape[1])
    return model


def _make_mock_tokenizer() -> MagicMock:
    """
    Mock a tokenizer whose __call__ returns a dict-like BatchEncoding.

    enc["input_ids"] → Tensor of shape [1, 3]
    enc.to(device) → enc (identity)
    tokenizer.decode(...) → "hello world"
    """
    tok = MagicMock()
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    tok.padding_side = "left"

    enc = MagicMock()
    enc.__getitem__ = MagicMock(
        side_effect=lambda k: (
            torch.tensor([PROMPT_INPUT_IDS]) if k == "input_ids"
            else torch.tensor([[1] * len(PROMPT_INPUT_IDS)])
        )
    )
    enc.to = MagicMock(return_value=enc)
    tok.return_value = enc
    tok.decode = MagicMock(return_value="hello world")
    return tok


@pytest.fixture
def engine() -> SpecdecEngine:
    """Fresh SpecdecEngine with mocked models for each test."""
    return SpecdecEngine(
        draft_model=_make_mock_model(),
        draft_tokenizer=_make_mock_tokenizer(),
        target_model=_make_mock_model(),
        target_tokenizer=_make_mock_tokenizer(),
        device=torch.device("cpu"),
        gamma=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpecdecEngineInit:
    def test_specdeckengine_init(self):
        """
        SpecdecEngine initialises without error when given mocked draft/target models.

        Checks that constructor stores gamma and device correctly, and that both
        model.eval() calls were made (required before inference).
        """
        device = torch.device("cpu")
        draft_model = _make_mock_model()
        target_model = _make_mock_model()

        eng = SpecdecEngine(
            draft_model=draft_model,
            draft_tokenizer=_make_mock_tokenizer(),
            target_model=target_model,
            target_tokenizer=_make_mock_tokenizer(),
            device=device,
        )

        assert eng.gamma == 4, f"Expected default gamma=4, got {eng.gamma}"
        assert eng.device == device
        draft_model.eval.assert_called_once()
        target_model.eval.assert_called_once()


class TestGenerate:
    def test_generate_returns_required_keys(self, engine: SpecdecEngine):
        """
        generate() must return a dict containing all five required keys.

        Missing any key would break downstream benchmarks that index directly
        into the result dict.
        """
        result = engine.generate("hello", max_new_tokens=4)
        required = {
            "generated_text",
            "tokens_generated",
            "acceptance_rate",
            "time_seconds",
            "tokens_per_second",
        }
        missing = required - result.keys()
        assert not missing, f"generate() result missing keys: {missing}"

    def test_acceptance_rate_bounds(self, engine: SpecdecEngine):
        """
        acceptance_rate must be in [0.0, 1.0] for any run.

        A value outside [0, 1] would mean the per-step acceptance fraction was
        computed incorrectly — either dividing by zero or counting more accepted
        tokens than were proposed.
        """
        result = engine.generate("test prompt", max_new_tokens=6)
        rate = result["acceptance_rate"]
        assert 0.0 <= rate <= 1.0, (
            f"acceptance_rate={rate:.4f} is outside [0.0, 1.0]"
        )

    def test_tokens_generated_within_budget(self, engine: SpecdecEngine):
        """
        tokens_generated must not exceed max_new_tokens.

        The engine uses both accepted draft tokens and bonus tokens; without
        careful budget tracking it can overshoot the requested limit.
        """
        max_new = 5
        result = engine.generate("hi", max_new_tokens=max_new)
        assert result["tokens_generated"] <= max_new, (
            f"Generated {result['tokens_generated']} tokens, budget was {max_new}"
        )


class TestSavePath:
    def test_save_results_creates_bench_speculative_json(self):
        """
        Calling save_results("bench_speculative", ...) creates a timestamped JSON in results/.

        This exercises the same save path that bench_speculative.py uses, ensuring
        at least one bench_speculative_*.json file exists after the test suite runs.
        """
        from inferno.utils import RESULTS_DIR, save_results

        before = set(RESULTS_DIR.glob("bench_speculative_*.json"))
        save_results("bench_speculative", {
            "draft_model": "mock",
            "target_model": "mock",
            "gamma": 2,
            "acceptance_rate": 0.75,
            "tokens_per_second": 10.0,
        })
        after = set(RESULTS_DIR.glob("bench_speculative_*.json"))
        new_files = after - before
        assert len(new_files) == 1, (
            f"Expected exactly 1 new bench_speculative JSON, got {len(new_files)}"
        )
