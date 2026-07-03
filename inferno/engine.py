"""
engine.py — Continuous batching scheduler for Inferno.

Implements a scheduler that dynamically groups in-flight requests into batches,
inserting new sequences mid-generation to maximize GPU utilization.
Chunk size is a tunable parameter; throughput is measured across chunk-size sweeps.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferno.utils import get_logger, wall_time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_BATCH_SIZE = 4
DEFAULT_MAX_PREFILL_CHUNK_SIZE = 512

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """A single inference request submitted to the engine."""

    prompt: str
    max_new_tokens: int
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    arrival_time: float = field(default_factory=time.time)


@dataclass
class CompletedRequest:
    """Result returned when a request finishes generation."""

    request_id: str
    output_text: str
    latency_ms: float        # wall time from prefill start to last token
    tokens_generated: int    # number of new tokens produced


@dataclass
class SchedulerConfig:
    """Tunable parameters for the continuous batching scheduler."""

    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    max_prefill_chunk_size: int = DEFAULT_MAX_PREFILL_CHUNK_SIZE

    # TRADEOFF: max_batch_size — larger batches amortize forward-pass overhead
    # on GPU (kernel launch, memory transfers) so throughput rises. On CPU,
    # however, batch dimension adds memory pressure without benefiting from
    # SIMD width. Keep max_batch_size small (1–4) on CPU.

    # TRADEOFF: max_prefill_chunk_size — a long prefill monopolises the decode
    # slot for the duration of the forward pass, stalling existing sequences.
    # Smaller chunks interleave prefill and decode more finely (lower tail
    # latency for decode), but more forward passes per prompt increases overhead.


# ---------------------------------------------------------------------------
# Internal per-sequence state (not exposed to callers)
# ---------------------------------------------------------------------------

@dataclass
class _RunningSequence:
    """Live state for a single in-flight sequence."""

    request: Request
    input_ids: torch.Tensor          # shape [1, prompt_len]
    attention_mask: torch.Tensor     # shape [1, prompt_len], extended each step
    generated_ids: list[int]         # accumulates new token ids
    cache: Any                        # past_key_values returned by model.forward()
    start_time: float                 # wall_time() at prefill start
    tokens_generated: int = 0
    is_prefilled: bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """
    Greedy-decode scheduler that admits new requests without waiting for the
    current batch to finish — the "continuous" part of continuous batching.

    Design
    ------
    - waiting_queue: requests not yet started (FIFO).
    - running: at most max_batch_size sequences in-flight simultaneously.
    - Each step() call promotes waiting → running (with prefill), then runs
      exactly one decode step per already-running sequence, then evicts
      finished sequences.
    - run_until_complete() loops step() until both queues are empty.

    CPU note
    --------
    Each sequence is processed individually (batch_size=1 per forward pass).
    True batching across multiple in-flight sequences would require padding
    to the same length and is expensive on CPU without benefit.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: SchedulerConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cpu")

        self._waiting: list[Request] = []
        self._running: list[_RunningSequence] = []

        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: Request) -> None:
        """Add a request to the waiting queue. Thread-safe for single-threaded use."""
        self._waiting.append(request)
        logger.debug("Queued request %s", request.request_id)

    def step(self) -> list[CompletedRequest]:
        """
        One scheduling iteration: admit new requests, decode all running sequences,
        evict finished ones.

        Returns a (possibly empty) list of CompletedRequest for sequences that
        finished during this step.
        """
        self._admit_waiting()
        self._decode_running()
        return self._evict_finished()

    def run_until_complete(self) -> list[CompletedRequest]:
        """
        Step in a loop until all submitted requests have completed.

        Returns the full list of CompletedRequest in completion order.
        """
        all_completed: list[CompletedRequest] = []
        while self._waiting or self._running:
            all_completed.extend(self.step())
        return all_completed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _admit_waiting(self) -> None:
        """
        Promote up to (max_batch_size - current_running) waiting requests into
        the running list by running their prefill forward pass.
        """
        free_slots = self.config.max_batch_size - len(self._running)
        to_admit = self._waiting[:free_slots]
        self._waiting = self._waiting[free_slots:]

        for req in to_admit:
            seq = self._prefill(req)
            if seq is not None:
                self._running.append(seq)

    def _prefill(self, request: Request) -> Optional[_RunningSequence]:
        """
        Tokenize the prompt, run a full forward pass, capture the KV cache, and
        record the first generated token from the prefill logits.

        Returns a _RunningSequence ready for decode steps, or None on error.
        """
        enc = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prefill_chunk_size,
        ).to(self.device)

        start = wall_time()
        with torch.no_grad():
            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                use_cache=True,
            )

        # Greedy: take argmax of last token's logits as first new token
        first_token_id: int = outputs.logits[:, -1, :].argmax(dim=-1).item()

        # Extend attention mask to cover the first generated token
        new_mask = torch.cat(
            [enc["attention_mask"], torch.ones(1, 1, device=self.device)],
            dim=1,
        )

        seq = _RunningSequence(
            request=request,
            input_ids=enc["input_ids"],
            attention_mask=new_mask,
            generated_ids=[first_token_id],
            cache=outputs.past_key_values,
            start_time=start,
            tokens_generated=1,
            is_prefilled=True,
        )
        logger.debug("Prefilled request %s (%d prompt tokens)", request.request_id, enc["input_ids"].shape[1])
        return seq

    def _decode_running(self) -> None:
        """
        Run exactly one decode step for each currently running sequence.

        Each sequence gets its own forward pass (batch_size=1) so that different
        sequence lengths don't force padding between them.

        # BASELINE: processing sequences one at a time — on GPU, true continuous
        # batching would pack multiple sequences into one forward pass using
        # variable-length (FlashAttention) or padded batching.
        """
        for seq in self._running:
            if not seq.is_prefilled:
                continue   # safety guard; all running seqs are prefilled

            last_token = torch.tensor([[seq.generated_ids[-1]]], device=self.device)
            # Extend mask: the new token is always non-padding (value=1)
            seq.attention_mask = torch.cat(
                [seq.attention_mask, torch.ones(1, 1, device=self.device)],
                dim=1,
            )

            with torch.no_grad():
                outputs = self.model(
                    input_ids=last_token,
                    attention_mask=seq.attention_mask,
                    past_key_values=seq.cache,
                    use_cache=True,
                )

            next_token_id: int = outputs.logits[:, -1, :].argmax(dim=-1).item()
            seq.generated_ids.append(next_token_id)
            seq.cache = outputs.past_key_values
            seq.tokens_generated += 1

    def _evict_finished(self) -> list[CompletedRequest]:
        """
        Remove sequences that have hit EOS or max_new_tokens from the running
        list and package them as CompletedRequest.
        """
        eos_id: int = self.tokenizer.eos_token_id or -1
        completed: list[CompletedRequest] = []
        still_running: list[_RunningSequence] = []

        for seq in self._running:
            last_id = seq.generated_ids[-1]
            is_eos = last_id == eos_id
            is_max = seq.tokens_generated >= seq.request.max_new_tokens

            if is_eos or is_max:
                # Trim trailing EOS before decoding
                ids_to_decode = seq.generated_ids
                if ids_to_decode and ids_to_decode[-1] == eos_id:
                    ids_to_decode = ids_to_decode[:-1]

                output_text = self.tokenizer.decode(ids_to_decode, skip_special_tokens=True)
                latency_ms = (wall_time() - seq.start_time) * 1000.0

                completed.append(CompletedRequest(
                    request_id=seq.request.request_id,
                    output_text=output_text,
                    latency_ms=latency_ms,
                    tokens_generated=seq.tokens_generated,
                ))
                logger.debug(
                    "Completed request %s — %d tokens, %.1f ms",
                    seq.request.request_id, seq.tokens_generated, latency_ms,
                )
            else:
                still_running.append(seq)

        self._running = still_running
        return completed
