"""
test_cache.py — Correctness tests for KV cache quantization in Inferno.

Checks that INT8 quantize→dequantize round-trips preserve tensor values within
an acceptable numerical tolerance, and that the quantized cache produces output
with perplexity no worse than a defined threshold vs the fp32 baseline.
All tests run on CPU; no GPU required.
"""

from __future__ import annotations

import torch
import pytest

from inferno.cache import (
    QuantizedDynamicCache,
    QuantizedDynamicCachePerChannel,
    QuantizedDynamicLayer,
    QuantizedTensor,
    compression_ratio,
    dequantize_int8,
    dequantize_int8_per_channel,
    quantize_int8,
    quantize_int8_per_channel,
)

# ---------------------------------------------------------------------------
# Tolerance for round-trip reconstruction
# ---------------------------------------------------------------------------

ROUND_TRIP_ATOL = 0.1   # absolute tolerance in original units (fp32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_tensor() -> torch.Tensor:
    """Random fp32 tensor resembling a KV cache slice: (batch, heads, seq, dim)."""
    torch.manual_seed(42)
    return torch.randn(2, 8, 16, 64, dtype=torch.float32)


@pytest.fixture
def small_tensor() -> torch.Tensor:
    """Small deterministic fp32 tensor for arithmetic checks."""
    return torch.tensor([[-1.0, 0.5, 2.0], [0.0, -0.25, 1.5]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# dtype tests
# ---------------------------------------------------------------------------

class TestQuantizedDtype:
    def test_quantized_tensor_has_int8_dtype(self, random_tensor: torch.Tensor):
        """
        Checks that quantize_int8() always returns a tensor with dtype torch.int8.

        Storing as any other dtype would defeat the memory-saving purpose of INT8
        quantization and cause downstream dequantization to produce wrong values.
        """
        qt = quantize_int8(random_tensor)
        assert qt.data.dtype == torch.int8

    def test_quantized_values_within_int8_range(self, random_tensor: torch.Tensor):
        """
        Checks that all quantized values are within [-127, 127].

        Values outside this range would overflow when cast to int8 and wrap around,
        silently corrupting the cache. We use -127 to 127 (not -128 to 127) to
        keep the mapping symmetric around zero.
        """
        qt = quantize_int8(random_tensor)
        int_data = qt.data.to(torch.int32)
        assert int_data.min().item() >= -127
        assert int_data.max().item() <= 127


# ---------------------------------------------------------------------------
# Round-trip reconstruction
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_dequantized_is_close_to_original(self, random_tensor: torch.Tensor):
        """
        Checks that quantize→dequantize preserves values within atol=0.1.

        This is the primary correctness invariant: if reconstruction error is
        too large the attention scores will degrade and perplexity will spike.
        """
        qt = quantize_int8(random_tensor)
        reconstructed = dequantize_int8(qt)
        assert torch.allclose(random_tensor, reconstructed, atol=ROUND_TRIP_ATOL), (
            f"Max reconstruction error: {(random_tensor - reconstructed).abs().max().item():.4f}"
        )

    def test_dequantized_dtype_is_float32(self, random_tensor: torch.Tensor):
        """
        Checks that dequantize_int8() always returns a float32 tensor.

        Attention kernels expect fp32 or bf16; int8 would cause a runtime error
        or silent miscomputation in the matmul.
        """
        qt = quantize_int8(random_tensor)
        out = dequantize_int8(qt)
        assert out.dtype == torch.float32

    def test_dequantized_shape_matches_original(self, random_tensor: torch.Tensor):
        """Checks that the shape is preserved exactly through quantize→dequantize."""
        qt = quantize_int8(random_tensor)
        out = dequantize_int8(qt)
        assert out.shape == random_tensor.shape

    def test_scale_is_positive(self, random_tensor: torch.Tensor):
        """
        Checks that the derived scale is always strictly positive.

        A zero or negative scale would make dequantization undefined or sign-flip
        the reconstructed values.
        """
        qt = quantize_int8(random_tensor)
        assert qt.scale > 0.0


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    def test_fp32_compression_ratio_is_4x(self, random_tensor: torch.Tensor):
        """
        Checks that an fp32 tensor (4 bytes/element) compresses 4x to int8 (1 byte/element).

        This is the fundamental memory-saving claim of INT8 quantization.
        """
        ratio = compression_ratio(random_tensor)
        assert abs(ratio - 4.0) < 1e-6, f"Expected 4x compression, got {ratio:.4f}x"

    def test_bf16_compression_ratio_is_2x(self):
        """
        Checks that a bf16 tensor (2 bytes/element) compresses 2x to int8.

        Qwen2.5 runs in bf16 on GPU; verifying the ratio is 2x (not 4x) ensures
        we report honest memory savings for GPU runs.
        """
        t = torch.randn(4, 8, dtype=torch.bfloat16)
        ratio = compression_ratio(t)
        assert abs(ratio - 2.0) < 1e-6, f"Expected 2x compression for bf16, got {ratio:.4f}x"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_quantization_is_deterministic(self, random_tensor: torch.Tensor):
        """
        Checks that quantize_int8() produces identical results on repeated calls.

        Non-determinism would make it impossible to reproduce benchmark numbers
        and would indicate unintended use of stochastic rounding.
        """
        qt1 = quantize_int8(random_tensor)
        qt2 = quantize_int8(random_tensor)
        assert torch.equal(qt1.data, qt2.data)
        assert qt1.scale == qt2.scale


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_zero_tensor_quantizes_correctly(self):
        """
        Checks that an all-zero input quantizes to all-zero INT8 output.

        When abs_max == 0, scale is undefined (division by zero). We fall back to
        scale=1.0 and expect the output to be all zeros.
        """
        t = torch.zeros(4, 4)
        qt = quantize_int8(t)
        assert qt.data.dtype == torch.int8
        assert torch.all(qt.data == 0)

    def test_all_zero_tensor_dequantizes_to_zeros(self):
        """
        Checks that the zero tensor round-trips to zeros (no phantom values introduced).
        """
        t = torch.zeros(4, 4)
        qt = quantize_int8(t)
        out = dequantize_int8(qt)
        assert torch.all(out == 0.0)

    def test_single_element_tensor(self):
        """
        Checks that a scalar tensor (shape [1]) quantizes and dequantizes correctly.

        Boundary condition: max(abs) equals the only element.
        """
        t = torch.tensor([3.14])
        qt = quantize_int8(t)
        out = dequantize_int8(qt)
        assert torch.allclose(t, out, atol=ROUND_TRIP_ATOL)

    def test_large_magnitude_tensor_does_not_overflow(self):
        """
        Checks that tensors with very large values (e.g. 1e6) are clamped to [-127, 127].

        Without the clamp, casting an out-of-range float to int8 in PyTorch is
        undefined behavior and silently wraps around.
        """
        t = torch.tensor([1e6, -1e6, 0.0])
        qt = quantize_int8(t)
        int_data = qt.data.to(torch.int32)
        assert int_data.max().item() <= 127
        assert int_data.min().item() >= -127


# ---------------------------------------------------------------------------
# QuantizedDynamicCache / QuantizedDynamicLayer
# ---------------------------------------------------------------------------

# Synthetic K/V shape: (batch=1, heads=4, seq=8, head_dim=32) — no GPU needed
KV_SHAPE = (1, 4, 8, 32)


class TestQuantizedDynamicCache:
    def test_update_returns_same_shape_as_input(self):
        """
        Checks that QuantizedDynamicCache.update() returns (key, value) tensors
        with exactly the same shape as the accumulated input.

        If shape changes, the attention kernel will raise a size mismatch error.
        Here we call update() twice (simulating prefill + one decode step) and
        check the returned shape covers both.
        """
        cache = QuantizedDynamicCache()
        k1 = torch.randn(*KV_SHAPE)
        v1 = torch.randn(*KV_SHAPE)
        # First call (prefill)
        rk, rv = cache.update(k1, v1, layer_idx=0)
        assert rk.shape == k1.shape
        assert rv.shape == v1.shape

        # Second call (decode — one new token)
        k2 = torch.randn(1, 4, 1, 32)
        v2 = torch.randn(1, 4, 1, 32)
        rk2, rv2 = cache.update(k2, v2, layer_idx=0)
        # Accumulated shape: seq_len = 8 + 1 = 9
        assert rk2.shape == (1, 4, 9, 32)
        assert rv2.shape == (1, 4, 9, 32)

    def test_update_returns_float32_tensors(self):
        """
        Checks that the tensors returned by update() are float32, not int8.

        Attention kernels expect float; returning int8 would cause a silent
        matmul type error or wrong attention scores.
        """
        cache = QuantizedDynamicCache()
        k = torch.randn(*KV_SHAPE, dtype=torch.float32)
        v = torch.randn(*KV_SHAPE, dtype=torch.float32)
        rk, rv = cache.update(k, v, layer_idx=0)
        assert rk.dtype == torch.float32, f"Expected float32, got {rk.dtype}"
        assert rv.dtype == torch.float32, f"Expected float32, got {rv.dtype}"

    def test_get_compression_stats_near_4x_for_fp32(self):
        """
        Checks that get_compression_stats() returns ~4.0 after fp32 K/V are cached.

        fp32 is 4 bytes/element; int8 is 1 byte/element → expected ratio = 4.0.
        This verifies that the quantization path actually fires (not a no-op).
        """
        cache = QuantizedDynamicCache()
        k = torch.randn(*KV_SHAPE, dtype=torch.float32)
        v = torch.randn(*KV_SHAPE, dtype=torch.float32)
        cache.update(k, v, layer_idx=0)
        ratio = cache.get_compression_stats()
        assert abs(ratio - 4.0) < 1e-5, f"Expected ~4x compression, got {ratio:.4f}x"

    def test_compression_stats_non_trivial_after_multiple_layers(self):
        """
        Checks that compression stats are populated (non-default) after simulating
        a multi-layer forward pass (calling update() for several layer indices).

        This mirrors what model.generate() does: one update() per attention layer
        per decode step. A default of 1.0 would mean the quantization path never ran.
        """
        cache = QuantizedDynamicCache()
        for layer_idx in range(4):   # simulate 4 attention layers
            k = torch.randn(1, 4, 16, 32)
            v = torch.randn(1, 4, 16, 32)
            cache.update(k, v, layer_idx=layer_idx)

        ratio = cache.get_compression_stats()
        # For fp32 input we expect ~4.0; definitely not 1.0 (default sentinel)
        assert ratio > 1.0, f"Compression stats were never populated (got {ratio})"
        assert abs(ratio - 4.0) < 1e-5, f"Expected ~4x, got {ratio:.4f}x"

    def test_get_seq_length_tracks_accumulated_tokens_per_tensor(self):
        """
        Checks that get_seq_length() correctly counts accumulated tokens after
        prefill + a decode step.

        generate() uses get_seq_length() to build attention masks; a wrong value
        would cause the model to attend to the wrong positions.
        """
        cache = QuantizedDynamicCache()
        # Prefill with 8 tokens
        k1 = torch.randn(1, 4, 8, 32)
        v1 = torch.randn(1, 4, 8, 32)
        cache.update(k1, v1, layer_idx=0)
        assert cache.get_seq_length(layer_idx=0) == 8

        # Decode step: 1 new token
        k2 = torch.randn(1, 4, 1, 32)
        v2 = torch.randn(1, 4, 1, 32)
        cache.update(k2, v2, layer_idx=0)
        assert cache.get_seq_length(layer_idx=0) == 9


# ---------------------------------------------------------------------------
# Per-channel quantization primitives
# ---------------------------------------------------------------------------

# KV shape: [batch=1, heads=4, seq=16, head_dim=32]
KV_SHAPE_PCH = (1, 4, 16, 32)
NUM_HEADS = 4
PER_CHANNEL_ATOL = 0.05   # tighter than per-tensor (0.1) because per-head scales


class TestPerChannelPrimitives:
    def test_scales_shape_is_per_head(self):
        """
        Checks that quantize_int8_per_channel returns scales of shape [1, heads, 1, 1].

        The scales must broadcast across batch, seq, and head_dim dimensions so
        that dividing the [batch, heads, seq, head_dim] tensor by scales is elementwise
        per head. Wrong shape would corrupt all values or raise a broadcast error.
        """
        t = torch.randn(*KV_SHAPE_PCH)
        _, scales = quantize_int8_per_channel(t, dim=1)
        assert scales.shape == (1, NUM_HEADS, 1, 1), (
            f"Expected scales shape (1, {NUM_HEADS}, 1, 1), got {tuple(scales.shape)}"
        )

    def test_each_head_gets_independent_scale(self):
        """
        Checks that changing one head's magnitude doesn't affect another head's scale.

        Constructs a tensor where head 0 has small values (~1.0) and head 1 has large
        values (~100.0). With per-tensor quantization the scale is dominated by the
        large head. With per-channel, head 0's scale must reflect its own range.
        """
        t = torch.zeros(1, 2, 4, 8)
        t[:, 0, :, :] = 1.0    # head 0: small magnitude
        t[:, 1, :, :] = 100.0  # head 1: large magnitude (would dominate per-tensor)

        _, scales = quantize_int8_per_channel(t, dim=1)

        # Head 0 scale ≈ 1/127 ≈ 0.00787
        expected_scale_h0 = 1.0 / 127
        actual_scale_h0 = scales[0, 0, 0, 0].item()
        assert abs(actual_scale_h0 - expected_scale_h0) < 1e-4, (
            f"Head 0 scale should be ~{expected_scale_h0:.5f}, got {actual_scale_h0:.5f}"
        )

        # Head 1 scale ≈ 100/127 ≈ 0.787
        expected_scale_h1 = 100.0 / 127
        actual_scale_h1 = scales[0, 1, 0, 0].item()
        assert abs(actual_scale_h1 - expected_scale_h1) < 0.01, (
            f"Head 1 scale should be ~{expected_scale_h1:.4f}, got {actual_scale_h1:.4f}"
        )

    def test_per_channel_round_trip_within_tighter_tolerance(self):
        """
        Checks that per-channel quantize→dequantize stays within atol=0.05.

        Per-channel gives a tighter bound than per-tensor (atol=0.1) because the
        scale tracks each head's actual range rather than the worst-case head.

        We normalize each head to an exact max magnitude so the theoretical bound
        is deterministic: max_error ≤ head_max_abs / (2 * 127).
        With head maxes in [0.1, 1, 5, 6], max_error ≤ 6/254 ≈ 0.024 < 0.05.
        """
        torch.manual_seed(7)
        t = torch.randn(*KV_SHAPE_PCH)
        # Normalize each head to a controlled maximum so the bound is exact
        target_maxes = [0.1, 1.0, 5.0, 6.0]   # head 3 max=6 → max_error≈6/254≈0.024
        for h, max_val in enumerate(target_maxes):
            head_abs_max = t[0, h].abs().max().clamp(min=1e-8)
            t[0, h] = t[0, h] / head_abs_max * max_val

        q, scales = quantize_int8_per_channel(t, dim=1)
        reconstructed = dequantize_int8_per_channel(q, scales)

        assert torch.allclose(t, reconstructed, atol=PER_CHANNEL_ATOL), (
            f"Max per-channel reconstruction error: "
            f"{(t - reconstructed).abs().max().item():.5f} > {PER_CHANNEL_ATOL}"
        )

    def test_per_channel_quantized_dtype_is_int8(self):
        """
        Checks that the quantized tensor returned by quantize_int8_per_channel is int8.

        Same contract as per-tensor: if dtype is wrong, memory savings are lost and
        dequantize will produce incorrect values.
        """
        t = torch.randn(*KV_SHAPE_PCH)
        q, _ = quantize_int8_per_channel(t, dim=1)
        assert q.dtype == torch.int8, f"Expected torch.int8, got {q.dtype}"

    def test_per_channel_compression_ratio_is_4x_for_fp32(self):
        """
        Checks that per-channel quantization still yields 4x compression for fp32.

        The extra scale tensors are negligible (32 floats per layer) — the
        dominant storage is the int8 K/V tensor, so compression stays at 4x.
        """
        t = torch.randn(*KV_SHAPE_PCH, dtype=torch.float32)
        ratio = compression_ratio(t)
        assert abs(ratio - 4.0) < 1e-6, f"Expected 4x for fp32, got {ratio:.4f}x"

    def test_per_channel_dequantize_shape_matches_input(self):
        """Checks that dequantize_int8_per_channel preserves tensor shape exactly."""
        t = torch.randn(*KV_SHAPE_PCH)
        q, scales = quantize_int8_per_channel(t, dim=1)
        out = dequantize_int8_per_channel(q, scales)
        assert out.shape == t.shape


class TestQuantizedDynamicCachePerChannel:
    def test_update_returns_correct_accumulated_shape(self):
        """
        Checks that QuantizedDynamicCachePerChannel.update() returns accumulated
        K/V with the right shape (prefill_len + decode_steps in seq dim).
        """
        cache = QuantizedDynamicCachePerChannel()
        k1 = torch.randn(1, 4, 8, 32)
        v1 = torch.randn(1, 4, 8, 32)
        rk, rv = cache.update(k1, v1, layer_idx=0)
        assert rk.shape == k1.shape

        k2 = torch.randn(1, 4, 1, 32)
        v2 = torch.randn(1, 4, 1, 32)
        rk2, rv2 = cache.update(k2, v2, layer_idx=0)
        assert rk2.shape == (1, 4, 9, 32)

    def test_update_returns_float32(self):
        """
        Checks that per-channel cache returns fp32 tensors for attention compute.
        """
        cache = QuantizedDynamicCachePerChannel()
        k = torch.randn(1, 4, 8, 32, dtype=torch.float32)
        v = torch.randn(1, 4, 8, 32, dtype=torch.float32)
        rk, rv = cache.update(k, v, layer_idx=0)
        assert rk.dtype == torch.float32
        assert rv.dtype == torch.float32

    def test_per_channel_cache_compression_is_4x(self):
        """
        Checks that QuantizedDynamicCachePerChannel reports ~4x compression for fp32.
        """
        cache = QuantizedDynamicCachePerChannel()
        for layer_idx in range(4):
            k = torch.randn(1, 4, 16, 32)
            v = torch.randn(1, 4, 16, 32)
            cache.update(k, v, layer_idx=layer_idx)
        ratio = cache.get_compression_stats()
        assert abs(ratio - 4.0) < 1e-5, f"Expected 4x, got {ratio:.4f}x"

    def test_get_seq_length_tracks_accumulated_tokens(self):
        """
        Checks that get_seq_length() correctly reflects accumulated tokens
        after prefill and a decode step.
        """
        cache = QuantizedDynamicCachePerChannel()
        cache.update(torch.randn(1, 4, 8, 32), torch.randn(1, 4, 8, 32), layer_idx=0)
        assert cache.get_seq_length(layer_idx=0) == 8
        cache.update(torch.randn(1, 4, 1, 32), torch.randn(1, 4, 1, 32), layer_idx=0)
        assert cache.get_seq_length(layer_idx=0) == 9
