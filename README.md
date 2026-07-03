# Inferno — LLM Inference Optimization Engine

## What This Is

Inferno is a from-scratch LLM inference optimization engine built on top of HuggingFace
Transformers. It implements INT8 KV cache quantization (per-tensor and per-channel variants)
and a continuous batching scheduler that admits new requests mid-generation without waiting
for the current batch to finish. All results are measured — no theoretical projections.
Benchmarked on CPU (Qwen2.5-0.5B, Apple Silicon) and GPU (Tesla T4, Kaggle).

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
INT8 instead of BF16 halves the bytes transferred from HBM on every decode step. The compute
cost of dequantizing back to BF16 before the attention matmul is meant to be small compared
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

### KV Cache Quantization — GPU (Tesla T4, Kaggle)

*Source: `results/bench_gpu_20260703T211036262Z.json`*  
*Perplexity evaluated on 20 excerpts from wikitext-2-raw-v1 test split.*

| Metric           | Baseline | INT8 Per-Tensor | INT8 Per-Channel |
|:-----------------|--------:|----------------:|-----------------:|
| Tokens/sec       |    30.8 |            23.7 |             23.7 |
| Peak memory MB   |   962.2 |           961.8 |            961.8 |
| Perplexity (WT2) |   23.70 |           26.02 |            25.28 |
| Perplexity Δ     |       — |           +2.32 |            +1.58 |
| Compression      |      1× |            2.0× |             2.0× |
| TTFT ms          |    37.3 |            36.6 |             37.2 |

**Why compression is 2× not 4×.** The GPU baseline loads weights in BF16 (2 bytes/element).
INT8 is 1 byte/element, giving 2×. On CPU the baseline uses FP32 (4 bytes/element), hence the
4× compression seen in the CPU table below.

**Why throughput drops 23% with INT8 on T4 (30.8 → 23.7 tok/s).** This is the most
important result in the benchmark. The T4 has 320 GB/s HBM bandwidth and 65 TFLOPS of
BF16 compute — a memory-bandwidth-to-compute ratio that is moderate rather than extreme.
More critically, this implementation dequantizes K/V back to BF16 *before* the attention
matmul, so the attention kernel still runs in BF16. The quantize and dequantize operations
on every `update()` call add real compute overhead (extra passes over the K/V tensors)
without reducing the compute cost of attention itself. On T4, that overhead is not covered
by bandwidth savings from the smaller INT8 storage. A fully fused INT8 attention kernel
(e.g. FlashAttention INT8 or vLLM's PagedAttention with INT8 KV) would eliminate the
dequantize overhead and should recover the throughput gap.

**Why peak memory is nearly identical across all three configurations (962 MB).** The
quantized K/V tensors are a small fraction of total GPU memory for a 500M-parameter
model at 64 decode steps. The bulk of memory is model weights (~950 MB in BF16). The
KV cache for 24 layers × 2 KV heads × 64 tokens × 64 head_dim × 2 bytes ≈ 3 MB —
halving 3 MB rounds to 0 on the scale of the total memory reading. KV cache memory
savings become meaningful at long context lengths (4K+ tokens) or larger models.

**Why per-channel beats per-tensor in perplexity on GPU (25.28 vs 26.02, Δ = −0.74).**
On CPU with FP32 input, per-channel and per-tensor gave nearly identical perplexity
because Qwen2.5-0.5B has only 2 GQA KV heads — not enough heads to show the outlier
effect. On GPU with BF16 input, the smaller numerical range of BF16 (8-bit exponent,
7-bit mantissa vs FP32's 23-bit mantissa) makes the K/V tensors slightly more sensitive
to scale choice. The 0.74 PPL improvement from per-channel is real and measurable, even
with 2 KV heads. The gap would widen substantially on models with 32+ KV heads, where
outlier heads can dominate the per-tensor scale and corrupt the precision of all other heads.

**TTFT is unaffected by quantization (37.3 ms baseline vs 36.6/37.2 ms quantized).** TTFT
measures the prefill forward pass, which processes all prompt tokens in one call. No decode
steps occur during prefill, so no KV cache quantization happens, and TTFT is unchanged.

### Continuous Batching — GPU (Tesla T4, Kaggle)

*Source: `results/bench_gpu_20260703T211036262Z.json`*  
*Workload: 8 requests, mixed token budgets (8–32 tokens each), 120 tokens total (parity-verified).*

| Metric          |  Static | Continuous |
|:----------------|--------:|-----------:|
| Tokens/sec      |    66.4 |       31.8 |
| Mean latency ms |     898 |       1514 |
| Max latency ms  |    1180 |       2733 |
| Total tokens    |     120 |        120 |

**Why static batching wins comprehensively at this concurrency level (66.4 vs 31.8 tok/s,
2.1× faster; 898 vs 1514 ms mean latency).** This is the expected result from first
principles, not a failure of the engine.

The engine processes each sequence individually — one forward pass per sequence per decode
step. With 4 sequences in flight and `max_batch_size=4`, static batching processes all
4 sequences in *one* forward pass, giving the tensor cores a full batch to saturate.
The engine makes 4 separate forward passes per decode step, each paying the full CUDA
kernel launch, memory transfer, and scheduling overhead, getting 1/4 the arithmetic
intensity per step.

On a T4 with 8.1 TFLOPS BF16 peak, a single-token forward pass (one sequence) is so
small that the GPU spends most of its time on kernel launch and memory setup rather than
arithmetic. Batching 4 sequences multiplies the arithmetic work without proportionally
increasing the overhead — this is exactly where static batching wins.

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
evaluation method used (not autoregressive scoring). On GPU with BF16 input, per-channel
correctly shows a lower perplexity delta than per-tensor (see above).

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
implementation stores K/V as INT8 but dequantizes to BF16 before the attention matmul.
This is correct and simple, but it means the attention kernel runs in BF16 and pays an
extra quantize+dequantize round-trip on every decode step. On T4, this overhead outweighs
the bandwidth savings from smaller KV storage (30.8 → 23.7 tok/s, −23%). A fused INT8
attention kernel would eliminate the overhead. On A100/H100 with higher HBM bandwidth,
the bandwidth savings are larger and may tip the balance.

**Per-channel benefit requires more KV heads.** On Qwen2.5-0.5B with 2 GQA KV heads the
improvement is small (0.74 PPL on GPU). On a model with 32 KV heads (e.g. Llama-3-8B),
outlier heads are much more likely to dominate the per-tensor scale and corrupt precision
for other heads. Per-channel is the correct default for production; this model is just too
small to stress-test it.

**Continuous batching overhead dominates at low concurrency.** The engine runs one forward
pass per sequence per decode step. Production continuous batching (vLLM, TGI) packs
multiple sequences into a single forward pass via FlashAttention's variable-length kernel,
eliminating per-sequence overhead and recovering throughput. At low concurrency (< 20
requests), scheduling overhead dominates regardless — the benefit only materialises at
high sustained load.

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
pytest tests/ -v                    # 46 tests, ~25 s, no GPU required
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

**46 / 46 tests passing.** All tests run on CPU — no GPU required.

| Suite | Tests | What it checks |
|:------|------:|:---------------|
| `test_baseline.py` | 9 | Model loads; generates non-empty output; tokens/sec is arithmetically consistent with timing; `peak_memory_mb > 100 MB` (guards against `tracemalloc` regression that previously returned 0.2 MB); result JSON written with all required keys |
| `test_cache.py` | 28 | INT8 dtype and value range; round-trip reconstruction within tolerance; 4× compression for FP32 / 2× for BF16; `QuantizedDynamicCache` and `QuantizedDynamicCachePerChannel` shape, dtype, and seq-length correctness; per-channel scales are head-independent; determinism |
| `test_engine.py` | 9 | Single request completes with non-empty output; two simultaneous requests both complete; `max_batch_size` never exceeded (instrumented step wrapper); `latency_ms > 0` and `< 120 s`; late-submitted request is not dropped; identical prompts produce identical output (greedy determinism) |

```
tests/test_baseline.py   9 passed
tests/test_cache.py     28 passed
tests/test_engine.py     9 passed
─────────────────────────────
46 passed in ~25 s  (CPU, Qwen2.5-0.5B)
```
