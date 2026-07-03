"""
cache.py — KV cache quantization for Inferno (INT8 first, INT4 later).

Implements quantization and dequantization of attention key/value tensors
to reduce memory footprint during inference. Correctness is validated by
comparing perplexity between quantized and fp32 caches within a tolerance bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from inferno.utils import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INT8_MAX = 127          # signed int8 range: -127 to 127 (we leave -128 unused)
BYTES_PER_FP32 = 4
BYTES_PER_INT8 = 1

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core quantization primitives
# ---------------------------------------------------------------------------

@dataclass
class QuantizedTensor:
    """Stores a quantized INT8 tensor together with its per-tensor scale."""
    data: torch.Tensor   # dtype=torch.int8
    scale: float         # dequantization multiplier: dequant = data * scale


def quantize_int8(tensor: torch.Tensor) -> QuantizedTensor:
    """
    Symmetric per-tensor INT8 quantization.

    Maps the full fp32/bf16 dynamic range onto [-127, 127] using a single
    scale factor derived from the absolute maximum value.

    # MATH: scale = max(|x|) / INT8_MAX  — maps largest magnitude to ±127
    # MATH: q = round(x / scale).clamp(-127, 127) — nearest-integer quantization
    """
    tensor_fp32 = tensor.float()
    abs_max = tensor_fp32.abs().max().item()

    if abs_max == 0.0:
        # Edge case: all-zero tensor — scale is undefined; use 1.0 to avoid div/0
        scale = 1.0
    else:
        # MATH: scale = max(|x|) / INT8_MAX — maps the largest magnitude to ±127
        scale = abs_max / INT8_MAX

    # MATH: q = round(x / scale).clamp(-127, 127) — nearest-neighbor rounding
    # then clamp to keep strictly within signed int8 range we use
    quantized = (tensor_fp32 / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)

    # GPU-NOTE: .to(torch.int8) preserves device; assert here catches any future
    # refactor that accidentally inserts a .cpu() call in the quantization path.
    assert quantized.device == tensor.device, (
        f"quantize_int8: device mismatch — input on {tensor.device}, "
        f"quantized on {quantized.device}"
    )

    # TRADEOFF: per-tensor scale is memory-cheap (one float per tensor) but
    # sacrifices accuracy vs per-channel or per-token scales because a single
    # outlier in any channel forces a coarse scale for the whole tensor.
    return QuantizedTensor(data=quantized, scale=scale)


def dequantize_int8(qt: QuantizedTensor, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Reconstruct an approximate fp32 tensor from an INT8 QuantizedTensor.

    # MATH: x_approx = q * scale — linear inverse of quantization
    """
    # MATH: x_approx = q * scale — multiply back by the scale to recover magnitudes
    return qt.data.to(torch.float32) * qt.scale


def compression_ratio(original: torch.Tensor) -> float:
    """
    Return bytes_before / bytes_after for FP32→INT8 quantization of this tensor.

    This is always BYTES_PER_FP32 / BYTES_PER_INT8 = 4x for any shape,
    but we compute it from the tensor's actual element size so bfloat16 inputs
    yield the correct 2x ratio rather than silently reporting 4x.
    """
    bytes_before = original.element_size() * original.numel()
    bytes_after = BYTES_PER_INT8 * original.numel()
    return bytes_before / bytes_after


# ---------------------------------------------------------------------------
# Per-channel quantization primitives
# ---------------------------------------------------------------------------

def quantize_int8_per_channel(
    tensor: torch.Tensor, dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-channel INT8 quantization: one scale per slice along `dim`.

    For KV cache tensors of shape [batch, heads, seq_len, head_dim], use dim=1
    to get one scale per attention head. This keeps each head's dynamic range
    independent, so an outlier head doesn't crush precision in well-behaved heads.

    # MATH: scale[h] = max(|tensor[:, h, :, :]|) / INT8_MAX
    #   — per-head max maps that head's largest magnitude to ±127
    # MATH: q[h] = round(tensor[:, h, :, :] / scale[h]).clamp(-127, 127)
    #   — same nearest-integer quantization as per-tensor, applied per head

    Returns:
        quantized: int8 tensor with same shape as input
        scales:    float32 tensor broadcastable to input shape
                   (e.g. [1, heads, 1, 1] for dim=1)
    """
    tensor_fp32 = tensor.float()
    ndim = tensor_fp32.ndim

    # Reduce over all dims except the channel dim to get per-channel abs max
    reduce_dims = tuple(i for i in range(ndim) if i != dim)
    # MATH: abs_max[c] = max(|tensor| with all dims except c reduced) — per-channel peak
    abs_max = tensor_fp32.abs().amax(dim=reduce_dims)          # shape: [num_channels]

    # Clamp to 1/INT8_MAX so an all-zero channel maps cleanly to all-zero int8
    abs_max = abs_max.clamp(min=1.0 / INT8_MAX)

    # MATH: scale[c] = abs_max[c] / INT8_MAX — maps each channel's peak to ±127
    scales_1d = abs_max / INT8_MAX                              # shape: [num_channels]

    # Reshape to broadcast against the original tensor
    scale_shape = [1] * ndim
    scale_shape[dim] = scales_1d.shape[0]
    scales = scales_1d.reshape(scale_shape).float()            # e.g. [1, heads, 1, 1]

    # MATH: q = round(x / scale).clamp(-127, 127) — per-channel division then rounding
    quantized = (tensor_fp32 / scales).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)

    # GPU-NOTE: scales is derived from tensor_fp32 (same device); quantized from scales.
    # Assert both stayed on the original device — catches accidental .cpu() in the path.
    assert quantized.device == tensor.device, (
        f"quantize_int8_per_channel: quantized on {quantized.device}, "
        f"expected {tensor.device}"
    )
    assert scales.device == tensor.device, (
        f"quantize_int8_per_channel: scales on {scales.device}, "
        f"expected {tensor.device}"
    )

    # TRADEOFF: per-channel scales cost more memory than per-tensor (one float per
    # head vs one float total), but the overhead is negligible — 32 heads × 2 floats
    # per K/V = 64 extra bytes per layer, vs MBs of int8 K/V tensor storage.
    return quantized, scales


def dequantize_int8_per_channel(
    quantized: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct fp32 from INT8 tensor and broadcastable per-channel scales.

    # MATH: x_approx[c] = q[c] * scale[c] — broadcast multiply restores magnitudes
    """
    # MATH: x_approx = q * scale — scales broadcast along all dims they are 1 in
    return quantized.to(torch.float32) * scales


# ---------------------------------------------------------------------------
# Quantized cache layer
# ---------------------------------------------------------------------------

class QuantizedDynamicLayer(DynamicLayer):
    """
    A DynamicLayer that stores accumulated K/V as INT8 instead of fp32/bf16.

    Replaces DynamicLayer's in-place tensor concatenation with a
    quantize→store→dequantize-on-read pattern. The returned K/V tensors
    are always in the original dtype so the attention kernel needs no changes.

    Storage: INT8 per tensor → 4x fewer bytes than fp32, 2x fewer than bf16.
    Compute: dequantized back to original dtype before attention matmuls.
    """

    def __init__(self) -> None:
        super().__init__()
        self._qt_keys: QuantizedTensor | None = None
        self._qt_values: QuantizedTensor | None = None
        self._compression_ratios: list[float] = []

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        """
        Record dtype/device for later use without allocating fp32 tensors.

        We override the parent to avoid creating the dummy empty fp32 tensors
        that DynamicLayer.lazy_initialization allocates; we store INT8 instead.
        """
        self.dtype = key_states.dtype
        self.device = key_states.device
        # Leave self.keys and self.values as None — we use _qt_keys/_qt_values
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V, quantize the accumulated cache as INT8, return dequantized.

        On the first call (prefill) key_states covers the full prompt length.
        On each decode step it covers exactly one new token.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        orig_dtype = key_states.dtype

        # Concatenate with any existing cache (dequantize stored INT8 first)
        if self._qt_keys is not None:
            prev_k = dequantize_int8(self._qt_keys).to(orig_dtype)
            prev_v = dequantize_int8(self._qt_values).to(orig_dtype)  # type: ignore[arg-type]
            full_k = torch.cat([prev_k, key_states], dim=-2)
            full_v = torch.cat([prev_v, value_states], dim=-2)
        else:
            full_k = key_states
            full_v = value_states

        # Track theoretical compression of the full accumulated K/V tensors
        self._compression_ratios.append(compression_ratio(full_k))
        self._compression_ratios.append(compression_ratio(full_v))

        # Quantize and store as INT8 — actual memory savings live here
        self._qt_keys = quantize_int8(full_k)
        self._qt_values = quantize_int8(full_v)

        # TRADEOFF: dequantizing before attention means compute happens in the
        # original dtype (correct) but memory savings only apply to storage, not
        # to the attention matmul itself. To save compute memory too, we would
        # need native INT8 attention kernels (e.g. Flash-Decoding INT8).
        return_k = dequantize_int8(self._qt_keys).to(orig_dtype)
        return_v = dequantize_int8(self._qt_values).to(orig_dtype)
        return return_k, return_v

    def get_seq_length(self) -> int:
        """Return the number of cached tokens (sequence length) for this layer."""
        if self._qt_keys is None:
            return 0
        return self._qt_keys.data.shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return (total_kv_length, kv_offset) needed for attention mask construction."""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        """QuantizedDynamicLayer is unbounded — returns -1 like DynamicLayer."""
        return -1

    def mean_compression_ratio(self) -> float:
        """Return mean compression ratio tracked across all update() calls."""
        if not self._compression_ratios:
            return 1.0
        return sum(self._compression_ratios) / len(self._compression_ratios)


# ---------------------------------------------------------------------------
# Quantized dynamic cache
# ---------------------------------------------------------------------------

class QuantizedDynamicCache(DynamicCache):
    """
    A drop-in replacement for DynamicCache that stores K/V tensors as INT8.

    Pass an instance to model.generate() via past_key_values to activate
    quantized caching with no other model changes required.

    Usage:
        cache = QuantizedDynamicCache()
        outputs = model.generate(**inputs, past_key_values=cache, ...)
        print(cache.get_compression_stats())  # ~4x for fp32, ~2x for bf16
    """

    def __init__(self) -> None:
        # Bypass DynamicCache.__init__ (which hard-codes DynamicLayer) and call
        # Cache.__init__ directly with our QuantizedDynamicLayer as the layer class.
        # This causes Cache.update() to lazily instantiate QuantizedDynamicLayer
        # instances when each layer is first accessed.
        Cache.__init__(self, layer_class_to_replicate=QuantizedDynamicLayer)

    def get_compression_stats(self) -> float:
        """
        Return mean compression ratio across all layers and all update() calls.

        For fp32 input this should be ~4.0; for bf16 input ~2.0.
        Returns 1.0 if no updates have occurred yet.
        """
        all_ratios: list[float] = []
        for layer in self.layers:
            if isinstance(layer, QuantizedDynamicLayer):
                all_ratios.extend(layer._compression_ratios)
        if not all_ratios:
            return 1.0
        return sum(all_ratios) / len(all_ratios)


# ---------------------------------------------------------------------------
# Per-channel quantized cache layer + cache
# ---------------------------------------------------------------------------

class QuantizedDynamicLayerPerChannel(DynamicLayer):
    """
    Like QuantizedDynamicLayer but uses per-channel (per-head) INT8 quantization.

    Stores one scale per attention head instead of one scale for the whole tensor.
    Reduces quantization error caused by magnitude imbalance across heads.

    Storage: int8 tensor + float32 scales [1, heads, 1, 1] per K/V buffer.
    The scale overhead is negligible vs the int8 tensor itself.
    """

    def __init__(self) -> None:
        super().__init__()
        self._keys_int8: torch.Tensor | None = None
        self._keys_scales: torch.Tensor | None = None
        self._values_int8: torch.Tensor | None = None
        self._values_scales: torch.Tensor | None = None
        self._compression_ratios: list[float] = []

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        """Record dtype/device without allocating placeholder fp32 tensors."""
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append K/V, re-quantize accumulated cache per-channel, return dequantized.

        dim=1 quantizes per attention head (shape [batch, heads, seq, head_dim]).
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        orig_dtype = key_states.dtype

        # Reconstruct previous accumulated cache if present
        if self._keys_int8 is not None:
            prev_k = dequantize_int8_per_channel(self._keys_int8, self._keys_scales).to(orig_dtype)  # type: ignore[arg-type]
            prev_v = dequantize_int8_per_channel(self._values_int8, self._values_scales).to(orig_dtype)  # type: ignore[arg-type]
            full_k = torch.cat([prev_k, key_states], dim=-2)
            full_v = torch.cat([prev_v, value_states], dim=-2)
        else:
            full_k = key_states
            full_v = value_states

        self._compression_ratios.append(compression_ratio(full_k))
        self._compression_ratios.append(compression_ratio(full_v))

        # dim=1 → one scale per attention head (the channel dimension in KV cache)
        self._keys_int8, self._keys_scales = quantize_int8_per_channel(full_k.float(), dim=1)
        self._values_int8, self._values_scales = quantize_int8_per_channel(full_v.float(), dim=1)

        # TRADEOFF: dequantizing before attention means compute happens in the
        # original dtype. Memory savings apply to storage only, not the matmul.
        return_k = dequantize_int8_per_channel(self._keys_int8, self._keys_scales).to(orig_dtype)
        return_v = dequantize_int8_per_channel(self._values_int8, self._values_scales).to(orig_dtype)
        return return_k, return_v

    def get_seq_length(self) -> int:
        """Return accumulated token count from the stored int8 key tensor."""
        if self._keys_int8 is None:
            return 0
        return self._keys_int8.shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return (total_kv_length, kv_offset) for attention mask construction."""
        query_length = cache_position.shape[0]
        return self.get_seq_length() + query_length, 0

    def get_max_cache_shape(self) -> int:
        """Unbounded dynamic cache — returns -1 like DynamicLayer."""
        return -1

    def mean_compression_ratio(self) -> float:
        """Return mean compression ratio across all update() calls."""
        if not self._compression_ratios:
            return 1.0
        return sum(self._compression_ratios) / len(self._compression_ratios)


class QuantizedDynamicCachePerChannel(DynamicCache):
    """
    Drop-in replacement for DynamicCache using per-channel INT8 quantization.

    One scale per attention head (vs one scale per whole K/V tensor in
    QuantizedDynamicCache). Lower quantization error at negligible extra cost.

    Usage:
        cache = QuantizedDynamicCachePerChannel()
        outputs = model.generate(**inputs, past_key_values=cache, ...)
    """

    def __init__(self) -> None:
        Cache.__init__(self, layer_class_to_replicate=QuantizedDynamicLayerPerChannel)

    def get_compression_stats(self) -> float:
        """Return mean compression ratio across all layers. ~4x for fp32, ~2x for bf16."""
        all_ratios: list[float] = []
        for layer in self.layers:
            if isinstance(layer, QuantizedDynamicLayerPerChannel):
                all_ratios.extend(layer._compression_ratios)
        if not all_ratios:
            return 1.0
        return sum(all_ratios) / len(all_ratios)


# ---------------------------------------------------------------------------
# Perplexity helper
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int = 256,
    cache: Optional[Cache] = None,
) -> float:
    """
    Compute mean token-level perplexity over a list of texts.

    Perplexity = exp(mean cross-entropy loss per token). Lower is better.
    Used to check that quantization does not meaningfully degrade model quality.

    If cache is provided (e.g. a QuantizedDynamicCache), a fresh instance is
    created for each text so the quantized K/V path is exercised per sequence.
    When cache=None, the model uses its default internal cache.
    """
    total_loss = 0.0
    total_batches = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            kwargs: dict = {"labels": enc["input_ids"]}
            if cache is not None:
                # Fresh cache per text so previous sequence's K/V doesn't bleed in.
                # type(cache)() works for QuantizedDynamicCache and
                # QuantizedDynamicCachePerChannel without branching.
                kwargs["past_key_values"] = type(cache)()
                kwargs["use_cache"] = True

            # labels=input_ids tells the model to compute cross-entropy loss
            out = model(**enc, **kwargs)
            total_loss += out.loss.item()
            total_batches += 1

    mean_loss = total_loss / total_batches if total_batches else float("inf")
    # MATH: perplexity = exp(mean_NLL) — converts nats of loss to a probability ratio
    return float(torch.exp(torch.tensor(mean_loss)).item())
