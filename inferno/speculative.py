"""
speculative.py — Speculative decoding engine for Inferno.

Implements the standard speculative sampling algorithm:
  1. Draft model proposes gamma tokens autoregressively, storing the full
     probability distribution at each step.
  2. Target model verifies all gamma positions in one forward pass.
  3. Each draft token is accepted with probability min(1, p_target / p_draft).
  4. On rejection, the next token is resampled from the corrected distribution
     max(0, p_target - p_draft) / Z, which guarantees the output distribution
     equals p_target exactly.

Reference: "Fast Inference from Transformers via Speculative Decoding"
           (Leviathan et al., 2023, arXiv:2211.17192)

Model pairing:
  Draft  — Qwen/Qwen2.5-0.5B-Instruct (same as DEFAULT_MODEL_ID in baseline.py)
  Target — Qwen/Qwen2.5-1.5B-Instruct (3x larger, better quality)
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferno.utils import get_logger, wall_time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_GAMMA = 4   # draft tokens proposed per speculative step

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SpecdecEngine:
    """
    Speculative decoding engine pairing a fast draft model with a slower target model.

    At each step the draft model samples gamma tokens autoregressively (cheap),
    then the target model verifies all gamma+1 positions in a single forward pass
    (amortised over gamma tokens instead of paying the target cost per token).
    The output distribution is provably identical to autoregressive sampling from
    the target model alone.

    acceptance_rate measures how often the draft matches the target: 1.0 means
    the draft predicts exactly what the target would have sampled (maximum speedup);
    0.0 means every draft token is rejected (no speedup, just overhead).
    """

    def __init__(
        self,
        draft_model: AutoModelForCausalLM,
        draft_tokenizer: AutoTokenizer,
        target_model: AutoModelForCausalLM,
        target_tokenizer: AutoTokenizer,
        device: torch.device,
        gamma: int = DEFAULT_GAMMA,
    ) -> None:
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.device = device
        self.gamma = gamma

        self.draft_model.eval()
        self.target_model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        """
        Generate up to max_new_tokens tokens using speculative decoding.

        Returns:
            generated_text     : decoded output string (EOS stripped)
            tokens_generated   : number of new tokens produced
            acceptance_rate    : mean fraction of draft tokens accepted per step (0–1)
            time_seconds       : total wall time
            tokens_per_second  : tokens_generated / time_seconds
        """
        enc = self.draft_tokenizer(prompt, return_tensors="pt").to(self.device)
        all_ids: torch.Tensor = enc["input_ids"]     # [1, prompt_len]

        generated_ids: list[int] = []
        acceptance_events: list[float] = []
        eos_id: int = self.draft_tokenizer.eos_token_id or -1

        t0 = wall_time()

        while len(generated_ids) < max_new_tokens:
            actual_gamma = min(self.gamma, max_new_tokens - len(generated_ids))

            # ------------------------------------------------------------------
            # Draft phase: sample gamma tokens from the draft model
            # ------------------------------------------------------------------
            draft_ids: list[int] = []
            draft_q_tok: list[float] = []        # draft prob of the chosen token
            draft_q_full: list[torch.Tensor] = []  # full vocab distribution at each step

            draft_past_kv = None
            cur_input = all_ids  # [1, current_len]; after first step becomes [1, 1]

            for _ in range(actual_gamma):
                with torch.no_grad():
                    draft_out = self.draft_model(
                        input_ids=cur_input,
                        past_key_values=draft_past_kv,
                        use_cache=True,
                    )

                # MATH: q(x) = softmax(logits[-1]) — draft distribution for next token
                logits = draft_out.logits[:, -1, :].float()   # [1, vocab]
                probs = torch.softmax(logits, dim=-1)          # [1, vocab]
                next_tok = torch.multinomial(probs, num_samples=1)  # [1, 1]
                tok_id = int(next_tok.item())

                draft_ids.append(tok_id)
                draft_q_tok.append(float(probs[0, tok_id].item()))
                draft_q_full.append(probs[0].cpu())

                draft_past_kv = draft_out.past_key_values
                cur_input = next_tok  # feed only the new token on subsequent steps

                if tok_id == eos_id:
                    break

            if not draft_ids:
                break

            # ------------------------------------------------------------------
            # Verification phase: one target forward pass over all gamma tokens
            # ------------------------------------------------------------------
            verify_input = torch.cat(
                [all_ids, torch.tensor([draft_ids], device=self.device)], dim=1
            )
            with torch.no_grad():
                target_out = self.target_model(input_ids=verify_input, use_cache=False)

            # Position (prefix_len - 1 + i) predicts draft_ids[i]
            prefix_len = all_ids.shape[1]

            # ------------------------------------------------------------------
            # Rejection sampling
            # ------------------------------------------------------------------
            newly_accepted: list[int] = []
            n_accepted = 0
            all_draft_accepted = True

            for i, (tok, q_tok, q_full) in enumerate(
                zip(draft_ids, draft_q_tok, draft_q_full)
            ):
                p_logits = target_out.logits[0, prefix_len - 1 + i, :].float()
                p_probs = torch.softmax(p_logits, dim=-1)   # [vocab]
                p_tok = float(p_probs[tok].item())

                # MATH: accept with prob min(1, p(x)/q(x)) — standard rejection sampling
                accept_prob = min(1.0, p_tok / max(q_tok, 1e-10))

                if float(torch.rand(1).item()) < accept_prob:
                    n_accepted += 1
                    newly_accepted.append(tok)
                    if tok == eos_id:
                        all_draft_accepted = False
                        break
                else:
                    # MATH: corrected distribution = max(0, p - q) / Z
                    # Sampling from this ensures the marginal output equals p exactly.
                    q_full_dev = q_full.to(self.device)
                    correction = torch.clamp(p_probs - q_full_dev, min=0.0)
                    z = correction.sum()
                    if z > 1e-10:
                        corrected_tok = int(torch.multinomial(correction / z, 1).item())
                    else:
                        corrected_tok = int(torch.multinomial(p_probs, 1).item())
                    newly_accepted.append(corrected_tok)
                    all_draft_accepted = False
                    break

            acceptance_events.append(n_accepted / len(draft_ids))

            if all_draft_accepted:
                # Bonus token: target distribution at position after all drafts
                bonus_pos = prefix_len - 1 + len(draft_ids)
                remaining_budget = max_new_tokens - len(generated_ids) - len(newly_accepted)
                if bonus_pos < target_out.logits.shape[1] and remaining_budget > 0:
                    bonus_logits = target_out.logits[0, bonus_pos, :].float()
                    bonus_probs = torch.softmax(bonus_logits, dim=-1)
                    bonus_tok = int(torch.multinomial(bonus_probs, 1).item())
                    newly_accepted.append(bonus_tok)

            # Extend running sequence
            generated_ids.extend(newly_accepted)
            if newly_accepted:
                all_ids = torch.cat(
                    [all_ids, torch.tensor([newly_accepted], device=self.device)],
                    dim=1,
                )

            if newly_accepted and newly_accepted[-1] == eos_id:
                break

        # Trim trailing EOS before decoding
        output_ids = generated_ids
        if output_ids and output_ids[-1] == eos_id:
            output_ids = output_ids[:-1]

        generated_text = self.draft_tokenizer.decode(output_ids, skip_special_tokens=True)
        elapsed = wall_time() - t0
        acceptance_rate = (
            sum(acceptance_events) / len(acceptance_events)
            if acceptance_events else 0.0
        )

        return {
            "generated_text": generated_text,
            "tokens_generated": len(generated_ids),
            "acceptance_rate": acceptance_rate,
            "time_seconds": elapsed,
            "tokens_per_second": len(generated_ids) / elapsed if elapsed > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Model loader helpers
# ---------------------------------------------------------------------------

def load_draft_and_target(
    device: Optional[torch.device] = None,
    draft_model_id: str = DRAFT_MODEL_ID,
    target_model_id: str = TARGET_MODEL_ID,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Load both draft and target models onto device.

    Returns (draft_model, draft_tokenizer, target_model, target_tokenizer, device).
    Uses bfloat16 on CUDA, float32 on CPU — consistent with load_model() in baseline.py.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    logger.info("Loading draft model: %s", draft_model_id)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id, trust_remote_code=True)
    if draft_tokenizer.pad_token is None:
        draft_tokenizer.pad_token = draft_tokenizer.eos_token
    draft_tokenizer.padding_side = "left"

    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id, dtype=dtype, trust_remote_code=True
    ).to(device)
    draft_model.eval()

    logger.info("Loading target model: %s", target_model_id)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_id, trust_remote_code=True)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    target_tokenizer.padding_side = "left"

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id, dtype=dtype, trust_remote_code=True
    ).to(device)
    target_model.eval()

    logger.info(
        "Draft params: %dM  Target params: %dM",
        sum(p.numel() for p in draft_model.parameters()) // 1_000_000,
        sum(p.numel() for p in target_model.parameters()) // 1_000_000,
    )
    return draft_model, draft_tokenizer, target_model, target_tokenizer, device
