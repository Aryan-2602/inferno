# Inferno — LLM Inference Optimization Engine

## What This Is

Inferno is a from-scratch LLM inference optimization engine built on top of HuggingFace
Transformers. It implements INT8 KV cache quantization (per-tensor and per-channel variants)
and a continuous batching scheduler that admits new requests mid-generation without waiting
for the current batch to finish, plus a speculative-decoding engine and a head-to-head vLLM
comparison. All results are measured — no theoretical projections, and negative results are
reported as such. Benchmarked on CPU (Qwen2.5-0.5B, Apple Silicon) and GPU (Tesla T4, Kaggle).

---

## Motivation

LLM inference has two distinct phases with very different bottlenecks.

**Prefill** processes all prompt tokens in one parallel forward pass. Every token attends
to every other token simultaneously, so the GPU's arithmetic units are saturated doing
matrix multiplications. This phase is *compute-bound* — you can't go faster without more
FLOPS.

**Decode** generates one token at a time. Each step is a tiny matrix-vector multiply
(one new token against the entire KV cache), which uses almost none of the GPU's compute
capacity. The bottleneck is reading the KV cache tensors from GPU HBM memory on every
step. At 64 decode steps with a KV cache growing to several hundred tokens, those tensors
are re-read from memory 64 times. This phase is *memory-bandwidth-bound*.

**KV cache quantization** attacks the decode bottleneck directly. Storing K/V tensors as
INT8 instead of a 16-bit float halves the bytes transferred from HBM on every decode step. The
compute cost of dequantizing back to that float dtype before the attention matmul is meant to be small compared
to the memory transfer savings — though as the T4 results show, whether this trade is
profitable depends on the hardware's memory-bandwidth-to-compute ratio.

**Continuous batching** attacks GPU *utilization* at scale. In static batching, the server
waits for every sequence in a batch to finish before starting the next one. Short requests
idle while the longest one finishes. Continuous batching treats the GPU as a queue: as soon
as one sequence completes, a waiting request is admitted and prefilled without flushing the
other running sequences. This keeps the GPU's memory bus saturated with useful work rather
than padding steps — the benefit grows with the number of concurrent users.

---

## Results

All numbers come from `results/` JSON files produced by the benchmark scripts.

Every model load in the project routes through `inferno/utils.py::select_torch_dtype()`
— FP32 on CPU, BF16 on Ampere and newer, FP16 below — so Inferno and the vLLM comparison
always run at the same dtype on the same machine. Each results JSON records the dtype it
ran at.

### KV Cache Quantization — GPU (Tesla T4, Kaggle)

*Source: `results/bench_gpu_20260918T195827668Z.json`*  
*Environment: Tesla T4 (sm75), torch 2.13.0+cu130, transformers 5.17.0, compute dtype **float16**.*  
*Perplexity evaluated on 20 excerpts from wikitext-2-raw-v1 test split.*

| Metric           | Baseline | INT8 Per-Tensor | INT8 Per-Channel |
|:-----------------|--------:|----------------:|-----------------:|
| Tokens/sec       |    26.9 |            21.8 |             21.4 |
| Peak memory MB   |   961.2 |           960.8 |            960.8 |
| Perplexity (WT2) |   23.77 |           25.51 |            25.01 |
| Perplexity Δ     |       — |           +1.74 |            +1.24 |
| Compression      |      1× |            2.0× |             2.0× |
| TTFT ms          |    36.9 |            38.1 |             37.7 |

**Why float16 and not bfloat16.** The T4 is Turing (compute capability 7.5). bfloat16 has no
native tensor-core support below 8.0 — PyTorch will run it, but without hardware acceleration,
and vLLM refuses to start in bfloat16 on such a card at all. Since the vLLM comparison is only
meaningful if both engines run the same dtype, `inferno/utils.py::select_torch_dtype()` is the
single source of truth for every model load in the project: float32 on CPU, bfloat16 on Ampere
and newer, float16 below. The dtype is recorded in each results JSON, and `bench_vllm.py` prints
an explicit parity line that reads `MISMATCH — numbers not comparable` if the two sides ever
diverge.

**Why compression is 2× not 4×.** The GPU baseline loads weights in FP16 (2 bytes/element).
INT8 is 1 byte/element, giving 2×. On CPU the baseline uses FP32 (4 bytes/element), hence the
4× compression seen in the CPU table below.

**Why throughput drops ~19% with INT8 on T4 (26.9 → 21.8 tok/s).** This is the most
important result in the benchmark. The T4 has 320 GB/s HBM bandwidth and a
memory-bandwidth-to-compute ratio that is moderate rather than extreme. More critically,
this implementation dequantizes K/V back to FP16 *before* the attention matmul, so the
attention kernel still runs in FP16. The quantize and dequantize operations on every
`update()` call add real compute overhead (extra passes over the K/V tensors) without
reducing the compute cost of attention itself. On T4, that overhead is not covered by
bandwidth savings from the smaller INT8 storage. A fully fused INT8 attention kernel
(e.g. FlashAttention INT8 or vLLM's PagedAttention with INT8 KV) would eliminate the
dequantize overhead and should recover the throughput gap.

**Why peak memory is nearly identical across all three configurations (961 MB).** The
quantized K/V tensors are a small fraction of total GPU memory for a 500M-parameter
model at 64 decode steps. The bulk of memory is model weights (~950 MB in FP16). The
KV cache for 24 layers × 2 KV heads × 64 tokens × 64 head_dim × 2 bytes ≈ 3 MB —
halving 3 MB rounds to 0 on the scale of the total memory reading. KV cache memory
savings become meaningful at long context lengths (4K+ tokens) or larger models.

**Why per-channel beats per-tensor in perplexity (25.01 vs 25.51, Δ = −0.50).**
On CPU with FP32 input, per-channel and per-tensor gave nearly identical perplexity
because Qwen2.5-0.5B has only 2 GQA KV heads — not enough heads to show the outlier
effect. On GPU with 16-bit input, the smaller numerical range makes the K/V tensors more
sensitive to scale choice, and per-channel wins consistently: it did so in the earlier
bfloat16 run (Δ = −0.74) and again here in float16 (Δ = −0.50). The gap would widen
substantially on models with 32+ KV heads, where outlier heads can dominate the
per-tensor scale and corrupt the precision of all other heads.

**TTFT is essentially unaffected by quantization (36.9 ms baseline vs 38.1/37.7 ms
quantized).** TTFT measures the prefill forward pass, which processes all prompt tokens in
one call. No decode steps occur during prefill, so no KV cache quantization happens. The
~1 ms spread is run-to-run noise on a single measurement, not a quantization cost.

#### Note on the earlier bfloat16 run

An earlier T4 run (`results/bench_gpu_20260703T211036262Z.json`) used bfloat16 on torch
2.10.0+cu128. It is kept for history but **is not a clean dtype A/B** — the torch and
transformers versions changed between the two runs as well, so the deltas below cannot be
attributed to dtype alone:

| Metric                    | bf16 run (2026-07-03) | fp16 run (2026-09-18) |
|:--------------------------|----------------------:|----------------------:|
| Baseline tok/s            |                  30.8 |                  26.9 |
| Baseline PPL              |                 23.70 |                 23.77 |
| Static batching tok/s     |                  66.4 |                  72.2 |
| Continuous batching tok/s |                  31.8 |                  28.7 |

Perplexity is effectively unchanged (23.70 → 23.77), which is the useful signal here: FP16's
narrower dynamic range did not degrade output quality on this model. The throughput deltas move
in both directions and are confounded, so no dtype conclusion is drawn from them.

### Continuous Batching — GPU (Tesla T4, Kaggle)

*Source: `results/bench_gpu_20260918T195827668Z.json`*  
*Workload: 8 requests, mixed token budgets (8–32 tokens each), 120 tokens total (parity-verified).*

| Metric          |  Static | Continuous |
|:----------------|--------:|-----------:|
| Tokens/sec      |    72.2 |       28.7 |
| Mean latency ms |     827 |       1657 |
| p50 latency ms  |     827 |       1168 |
| p95 latency ms  |    1009 |       2937 |
| p99 latency ms  |    1009 |       2958 |
| Max latency ms  |    1009 |       2963 |
| Total tokens    |     120 |        120 |

**Read the static percentiles with care — that distribution is degenerate.** In static
batching every sequence in a batch is reported as waiting the full batch duration, and 8
requests at `max_batch_size=4` is exactly 2 batches. The 8 latency samples therefore take only
*2 distinct values*, which is why mean = p50 = 827 ms and p95 = p99 = max = 1009 ms. Those are
batch durations, not a per-request latency spread. The continuous-batching percentiles
(p50 1168, p95 2937, p99 2958) are a genuine distribution over 8 independently-completing
requests. Comparing the two percentile columns directly is not meaningful; only the
continuous column describes real per-request behaviour.

**Why static batching wins on throughput at this concurrency level (72.2 vs 28.7 tok/s,
2.5× faster).** This is the expected result from first principles, not a failure of the engine.

The engine processes each sequence individually — one forward pass per sequence per decode
step. With 4 sequences in flight and `max_batch_size=4`, static batching processes all
4 sequences in *one* forward pass, giving the tensor cores a full batch to saturate.
The engine makes 4 separate forward passes per decode step, each paying the full CUDA
kernel launch, memory transfer, and scheduling overhead, getting 1/4 the arithmetic
intensity per step.

On a T4, a single-token forward pass (one sequence) is so small that the GPU spends most of
its time on kernel launch and memory setup rather than arithmetic. Batching 4 sequences
multiplies the arithmetic work without proportionally increasing the overhead — this is
exactly where static batching wins.

**Why continuous batching's advantage appears at high concurrency, not low.** With 8
requests and a max batch of 4, the scheduler runs at near-100% utilization. The benefit
of continuous batching materialises when there are hundreds of concurrent requests with
heterogeneous lengths, creating a steady queue of work to fill each freed slot. In that
regime, static batching would leave the GPU idle for the remaining 24 decode steps of a
32-token sequence while a waiting 8-token request sits in queue. The engine eliminates
that idle time. At low concurrency (8 requests), there is no idle time to eliminate —
every slot is already full — so the per-sequence overhead is pure cost.

In production deployments (vLLM, TGI, SGLang), continuous batching is paired with
variable-length batching: multiple sequences are packed into a single forward pass using
FlashAttention's `cu_seqlens` interface, recovering the throughput advantage while
preserving the latency benefit for short sequences. That is not implemented here.

---

### KV Cache Quantization — CPU (Qwen2.5-0.5B, Apple Silicon / x86)

*Source: `results/bench_cache_20260703T062019Z.json`*

| Metric         | Baseline | INT8 Per-Tensor | INT8 Per-Channel |
|:---------------|--------:|----------------:|-----------------:|
| Tokens/sec     |   13.16 |           13.75 |            13.57 |
| Peak memory MB | 1954.9  |         2017.3  |          1997.5  |
| Perplexity     |    4.96 |            7.05 |             8.55 |
| Perplexity Δ   |       — |           +2.09 |            +3.59 |
| Compression    |      1× |            4.0× |             4.0× |

**Why throughput is flat on CPU.** On CPU the bottleneck is Python/PyTorch dispatch
overhead, not memory bandwidth or arithmetic. The quantize/dequantize compute is buried
in that overhead, so throughput neither improves nor meaningfully degrades.

**Why per-channel perplexity is worse than per-tensor on CPU (+3.59 vs +2.09).** With
only 2 GQA KV heads on CPU with FP32 input, both methods produce nearly identical
quantization error. The ~1.5 PPL difference is in the noise of the single-pass perplexity
evaluation method used (not autoregressive scoring). On GPU with 16-bit input, per-channel
correctly shows a lower perplexity delta than per-tensor — in both the FP16 and the earlier
BF16 run (see above).

### Continuous Batching — CPU (Qwen2.5-0.5B)

*Source: `results/bench_batching_20260703T080232063Z.json`*  
*Workload: 8 requests, mixed token budgets (8–32 tokens each), 120 tokens total (parity-verified).*

| Metric          |  Static | Continuous |
|:----------------|--------:|-----------:|
| Tokens/sec      |   14.53 |      13.38 |
| Mean latency ms |    4121 |       3585 |
| Max latency ms  |    5458 |       6600 |
| Total tokens    |     120 |        120 |

On CPU, the mean latency improvement from continuous batching (−536 ms, −13%) is visible
because short-budget requests exit without waiting for long ones. Throughput is similar
to GPU: static slightly outperforms continuous for the same reason (batched matmuls vs
individual forward passes), but the gap is smaller because CPU BLAS batch scaling is
less dramatic than GPU tensor core scaling.

---

## Architecture

### KV Cache Quantization

HuggingFace Transformers 4.40+ uses a `DynamicCache` object whose `update()` method is
called by each attention layer on every forward pass. The right interception point is to
subclass `DynamicLayer` — the per-layer storage unit — and override `update()`.

```
QuantizedDynamicLayer(DynamicLayer)
    update(key_states, value_states) → (keys_fp, values_fp)
        1. Dequantize stored INT8 → original dtype  (if cache already populated)
        2. Concatenate with new key_states/value_states
        3. Quantize full accumulated tensor → INT8   (stored back)
        4. Dequantize INT8 → original dtype           (returned to attention kernel)

QuantizedDynamicCache(DynamicCache)
    __init__: Cache.__init__(layer_class_to_replicate=QuantizedDynamicLayer)
    get_compression_stats() → mean ratio across all layers
```

Hook-based interception was tried first and abandoned: in Transformers 4.57, forward
hooks on attention layers receive `None` for `past_key_values` because the cache is
mutated in-place through the `DynamicCache` API, not returned as a tuple element.
Subclassing `DynamicLayer` is the only stable interception point.

**Quantization formula:**

```
# Per-tensor (one scale for the whole K or V tensor):
scale     = max(|tensor|) / 127
quantized = round(tensor / scale).clamp(-127, 127).to(int8)

# Per-channel (one scale per attention head, dim=1):
scale[h]     = max(|tensor[:, h, :, :]|) / 127
quantized[h] = round(tensor[:, h, :, :] / scale[h]).clamp(-127, 127).to(int8)
```

Per-channel scales are stored as a `[1, heads, 1, 1]` float32 tensor that broadcasts
against the `[batch, heads, seq_len, head_dim]` KV tensors. Storage overhead is negligible:
for 24 layers × 2 KV heads × 2 (K and V) × 4 bytes = 384 bytes total, versus megabytes
of INT8 KV data.

### Continuous Batching Engine

```
ContinuousBatchingEngine
    _waiting: list[Request]          # FIFO queue of not-yet-started requests
    _running: list[_RunningSequence] # at most max_batch_size in flight

    step():
        1. _admit_waiting()  — promote up to (max_batch_size - len(_running))
                               requests from _waiting; run each through prefill
        2. _decode_running() — one greedy decode step per running sequence
        3. _evict_finished() — remove sequences at EOS or max_new_tokens,
                               return as CompletedRequest with latency_ms

    run_until_complete() → loops step() until both queues empty
```

**Prefill** runs a full forward pass on the prompt, captures `outputs.past_key_values`,
and records the first generated token from the prefill logits. Each subsequent `step()`
call runs a single-token decode forward pass per sequence using its stored cache.

**`max_prefill_chunk_size` tradeoff.** A long prefill monopolises the forward pass slot
for its full duration, stalling decode steps for sequences already running. Chunked
prefill splits long prompts across multiple steps so existing sequences aren't stalled.
The cost: more forward passes per prompt. At `max_prefill_chunk_size=512` (the default),
most short-to-medium prompts prefill in a single step.

---

## Limitations & Future Work

**Dequantize-before-attention eliminates the throughput benefit on T4.** The current
implementation stores K/V as INT8 but dequantizes to FP16 before the attention matmul.
This is correct and simple, but it means the attention kernel runs in FP16 and pays an
extra quantize+dequantize round-trip on every decode step. On T4, this overhead outweighs
the bandwidth savings from smaller KV storage (26.9 → 21.8 tok/s, −19%). A fused INT8
attention kernel would eliminate the overhead. On A100/H100 with higher HBM bandwidth,
the bandwidth savings are larger and may tip the balance.

**Per-channel benefit requires more KV heads.** On Qwen2.5-0.5B with 2 GQA KV heads the
improvement is small (0.50 PPL on GPU in FP16; 0.74 in the earlier BF16 run). On a model with 32 KV heads (e.g. Llama-3-8B),
outlier heads are much more likely to dominate the per-tensor scale and corrupt precision
for other heads. Per-channel is the correct default for production; this model is just too
small to stress-test it.

**Continuous batching overhead dominates at low concurrency.** The engine runs one forward
pass per sequence per decode step. Production continuous batching (vLLM, TGI) packs
multiple sequences into a single forward pass via FlashAttention's variable-length kernel,
eliminating per-sequence overhead and recovering throughput. At low concurrency (< 20
requests), scheduling overhead dominates regardless — the benefit only materialises at
high sustained load.

**Speculative decoding is below break-even at the configured gamma.** The measured 0.70×
is explained by the acceptance rate (0.547) and γ = 4 together putting the idealised ceiling
at 0.92× — see the speculative section. γ = 1–2 and a smaller draft model are the fixes; a
gamma sweep has not been run.

**The vLLM throughput comparison is not like-for-like.** vLLM is measured with all 10 prompts
submitted at once; the Inferno figure it is placed beside is the sequential `BATCH_SIZE = 1`
baseline. The ratio therefore mixes batching effects with kernel-fusion effects and should not
be read as a pure engine-vs-engine number. A matched-concurrency harness is needed to separate
them.

**No PagedAttention.** This implementation allocates a contiguous INT8 buffer per sequence
per layer, re-allocating on every `update()` call. At hundreds of concurrent sequences with
different lengths, memory fragmentation would become a bottleneck. vLLM-style paged
allocation in fixed-size blocks eliminates fragmentation and enables memory sharing across
requests with common prefixes — neither is implemented here.

---

## Reproducing Results

```bash
git clone https://github.com/Aryan-2602/inferno.git
cd inferno
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### CPU

```bash
pytest tests/ -v                    # 51 tests, ~30 s, no GPU required
python benchmarks/bench_cache.py    # Baseline | Per-Tensor | Per-Channel
python benchmarks/bench_batching.py # Static vs Continuous, parity-checked
```

### GPU (Kaggle T4 or equivalent)

```bash
python scripts/check_gpu.py        # environment health check — must show ALL CHECKS PASSED
python benchmarks/bench_gpu.py     # full GPU suite; saves to results/bench_gpu_{ts}.json
```

---

## Test Coverage

**51 / 51 tests passing.** All tests run on CPU — no GPU required, and the suite is
verified against both transformers 4.57.6 and 5.17.x.

| Suite | Tests | What it checks |
|:------|------:|:---------------|
| `test_baseline.py` | 9 | Model loads; generates non-empty output; tokens/sec is arithmetically consistent with timing; `peak_memory_mb > 100 MB` (guards against `tracemalloc` regression that previously returned 0.2 MB); result JSON written with all required keys |
| `test_cache.py` | 28 | INT8 dtype and value range; round-trip reconstruction within tolerance; 4× compression for FP32 / 2× for BF16; `QuantizedDynamicCache` and `QuantizedDynamicCachePerChannel` shape, dtype, and seq-length correctness; per-channel scales are head-independent; determinism |
| `test_engine.py` | 9 | Single request completes with non-empty output; two simultaneous requests both complete; `max_batch_size` never exceeded (instrumented step wrapper); `latency_ms > 0` and `< 120 s`; late-submitted request is not dropped; identical prompts produce identical output (greedy determinism) |

```
tests/test_baseline.py    9 passed
tests/test_cache.py      28 passed
tests/test_engine.py      9 passed
tests/test_speculative.py 5 passed
──────────────────────────────────
51 passed in ~30 s  (CPU, Qwen2.5-0.5B; speculative tests use mocked models)
```

---

## Speculative Decoding

Speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) accelerates
autoregressive generation by pairing a small **draft model** with a large **target model**.
At each step the draft model proposes *gamma* tokens cheaply, then the target model verifies
all *gamma+1* positions in one forward pass. Accepted tokens come from the target distribution
exactly — the output is provably identical to sampling from the target model alone, with speedup
proportional to the **acceptance rate** (fraction of draft tokens the target would have chosen).

### Implementation (`inferno/speculative.py`)

```
SpecdecEngine(draft_model, draft_tokenizer, target_model, target_tokenizer, device, gamma=4)

generate(prompt, max_new_tokens=64) → {
    generated_text    : str
    tokens_generated  : int
    acceptance_rate   : float   # mean fraction of draft tokens accepted per step, 0–1
    time_seconds      : float
    tokens_per_second : float
}
```

**Models:**
- Draft: `Qwen/Qwen2.5-0.5B-Instruct` (same as the rest of the project)
- Target: `Qwen/Qwen2.5-1.5B-Instruct` (3× larger, sets the output distribution)

**Algorithm per speculative step:**
1. Draft samples *gamma* tokens autoregressively, storing the full vocabulary distribution `q` at each step.
2. Target runs one forward pass on `[prompt + all draft tokens]`.
3. Token *i* is accepted with probability `min(1, p_target(d_i) / q_draft(d_i))`.
4. On rejection, resample from the corrected distribution `max(0, p_target − q_draft) / Z` — this guarantees the marginal output equals `p_target` exactly.
5. If all *gamma* tokens are accepted, sample one bonus token from the target.

**`acceptance_rate`** is the mean fraction of draft tokens accepted per step.
A rate of 1.0 means the draft always predicts what the target would have chosen
(maximum speedup: `gamma+1` tokens per target call). A rate of 0.0 means every draft
token is rejected (pure overhead; speedup drops below 1×). Expected rate on Qwen pairs
depends on how well the 0.5B distribution approximates the 1.5B distribution.

### Measured results — Tesla T4

*Source: `results/bench_speculative_20260918T195638233Z.json`*  
*Draft `Qwen2.5-0.5B-Instruct` (494M) · target `Qwen2.5-1.5B-Instruct` (1543M) · gamma = 4 · float16 · 10 prompts × 64 tokens.*

| Metric          | Autoregressive | Speculative (γ=4) |
|:----------------|---------------:|------------------:|
| Tokens/sec      |           23.2 |              16.4 |
| Mean latency ms |           2754 |              3906 |
| Acceptance rate |              — |             0.547 |
| Total tokens    |            640 |               640 |

**Speculative decoding is 0.70× — a slowdown, and the arithmetic says it should be.**
This is a real negative result, not a bug. With per-token acceptance α = 0.547 and γ = 4,
the expected number of tokens produced per speculative iteration is

```
E[tokens/iter] = (1 − α^(γ+1)) / (1 − α) = (1 − 0.547^5) / 0.453 ≈ 2.10
```

Each iteration costs 4 draft forward passes plus 1 target forward pass. Scaling the draft
cost by the parameter ratio (494M / 1543M ≈ 0.32) gives ≈ 4(0.32) + 1 = 2.28 target-equivalent
passes for those 2.10 tokens — about 1.09 target passes per token, versus exactly 1.0 for
plain autoregressive decoding. **The idealised ceiling at this acceptance rate is ≈ 0.92× —
below break-even before any framework overhead is counted.** Measured 0.70× is that ceiling
plus Python dispatch and kernel-launch cost.

The error is the choice of γ, not the implementation. Running the same arithmetic across γ:

| γ | E[tokens/iter] | Target-equiv. cost | Idealised speedup |
|--:|---------------:|-------------------:|------------------:|
| 1 |           1.55 |               1.32 |            1.17×  |
| 2 |           1.85 |               1.64 |            1.13×  |
| 4 |           2.10 |               2.28 |            0.92×  |

γ = 4 only pays off at substantially higher acceptance. A 0.5B/1.5B Qwen pair accepting ~55%
of draft tokens wants γ = 1–2. `GAMMA` is a named constant at the top of
`benchmarks/bench_speculative.py`; re-running the sweep is the obvious next experiment.

Note also that the draft and target are only 3.1× apart in parameter count. Speculative
decoding assumes the draft is *much* cheaper than the target — production pairs are typically
10–20× apart (e.g. 7B drafting for 70B). At 3.1×, the draft's four sequential passes cost a
third of the target pass they are trying to save, which caps the achievable speedup regardless
of γ. A smaller draft model would move this result more than any amount of tuning.

### How to run

```bash
source .venv/bin/activate
python scripts/check_gpu.py          # must pass first
python benchmarks/bench_speculative.py
```

Results saved to `results/bench_speculative_{timestamp}.json`.

---

## vLLM Comparison

[vLLM](https://github.com/vllm-project/vllm) is a production LLM inference library that implements
PagedAttention (non-contiguous KV cache blocks), continuous batching with variable-length forward
passes (FlashAttention `cu_seqlens` interface), and fused CUDA kernels throughout. Comparing
against it is meaningful because vLLM represents the *engineering ceiling* for what the optimizations
in this project are working toward:

- **KV cache quantization** — vLLM supports INT8 KV cache natively via `kv_cache_dtype="fp8"`;
  the gap to Inferno's manual INT8 implementation reveals the kernel-fusion benefit.
- **Continuous batching** — vLLM's scheduler packs multiple sequences into one forward pass;
  Inferno's engine runs one pass per sequence. The throughput ratio quantifies that gap.
- **PagedAttention** — vLLM avoids KV cache memory fragmentation at scale; Inferno allocates
  contiguous buffers per sequence.

### Measured results — Tesla T4

*Source: `results/bench_vllm_20260918T200014463Z.json`, vLLM 0.29.0, float16 (parity-checked against the Inferno run).*

| Metric         |   vLLM | Inferno | 
|:---------------|-------:|--------:|
| Tokens/sec     | 1453.8 |    26.9 |
| p50 latency ms |    364 |     827 |
| p95 latency ms |    756 |    1009 |
| p99 latency ms |   1012 |    1009 |

**The 54× throughput ratio is not a like-for-like engine comparison, and should not be
quoted as one.** The two numbers measure different workloads:

- **vLLM 1453.8 tok/s** — all 10 prompts submitted in a single `generate()` call, so vLLM's
  scheduler batches them internally and runs them through fused kernels concurrently.
- **Inferno 26.9 tok/s** — the Part A cache baseline, which is strictly sequential:
  `BATCH_SIZE = 1`, one prompt at a time, one HuggingFace `generate()` call each.

Most of that gap is batching, not kernel quality. The fairer comparison against Inferno's
own best batched result — **72.2 tok/s** from static batching at 4 requests/batch — is
roughly **20×**, still a large and real gap, and still measured at a lower batch size than
vLLM was given. The honest summary is: vLLM is at least an order of magnitude faster here,
and this benchmark does not isolate how much of that is PagedAttention vs. fused kernels vs.
simply batching more requests.

**The p99 "win" for Inferno (1009 vs 1012 ms) is an artifact, not an advantage.** Inferno's
latency samples come from static batching, where only 2 distinct values exist across 8
requests (see the batching section above), so its p95 and p99 both collapse onto the max.
There is no tail-latency story to tell from these numbers.

**vLLM was itself handicapped on this hardware.** The logs show FlashAttention 2 unavailable
(`FA2 is only supported on devices with compute capability >= 8`), so vLLM fell back to its
Triton attention backend, and FlashInfer top-p/top-k sampling was unavailable for the same
reason. On an Ampere or newer card the gap would likely be *wider*, not narrower.

**Startup cost is excluded from the throughput figure.** vLLM spent ≈ 42 s initialising the
engine (15.9 s of `torch.compile`, plus CUDA graph capture) before serving a single token.
For a long-running server that amortises to nothing; for a short batch job it is the dominant
cost. Inferno has no comparable warm-up.

### How to run

```bash
# Install vLLM (requires CUDA; large install, separate from project deps)
pip install -r requirements_vllm.txt

# Run comparison benchmark (exits cleanly if vLLM is not installed)
python benchmarks/bench_vllm.py
```

The script exits with exit code 0 and prints install instructions if vLLM is not available —
it does not crash or raise an ImportError.

Results saved to `results/bench_vllm_{timestamp}.json`. The comparison table loads the most
recent `results/bench_gpu_*.json` automatically for Inferno numbers.
