from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    decode_fp8_block_scaled_words,
    decode_marlin_fp8_block_words,
    dequantize_fp8_block_scaled,
    encode_fp8_block_scaled,
    encode_marlin_fp8_block,
    encoded_size,
    fp8_block_scale_geometry,
    marlin_fp8_block_geometry,
)


def _bf16_words(*words: int) -> torch.Tensor:
    signed = [word if word < 0x8000 else word - 0x10000 for word in words]
    return torch.tensor(signed, dtype=torch.int16).view(torch.bfloat16)


def test_block_scale_fp8_round_trip_and_partial_tile_dequantization():
    shape = (130, 129)
    geometry = fp8_block_scale_geometry("FP8_E4M3FN_BLOCK128_BF16S", shape)
    assert (geometry.m_tiles, geometry.k_tiles) == (2, 2)

    codes = torch.zeros(shape, dtype=torch.uint8)
    codes[:128, :128] = 0x38
    codes[:128, 128:] = 0x40
    codes[128:, :128] = 0x40
    codes[128:, 128:] = 0x40
    scales = _bf16_words(0x3F80, 0x4000, 0x4080, 0x3F00).reshape(2, 2)
    payload = encode_fp8_block_scaled(codes, scales, shape)

    assert len(payload) == encoded_size(
        "blockscale-m128-k128-v1", "FP8_E4M3FN_BLOCK128_BF16S", shape
    )
    assert payload[: geometry.code_plane_bytes] == codes.numpy().tobytes()
    assert not any(payload[geometry.code_plane_bytes : geometry.scale_plane_offset])
    assert payload[geometry.scale_plane_offset :] == struct.pack(
        "<HHHH", 0x3F80, 0x4000, 0x4080, 0x3F00
    )

    decoded_codes, decoded_scales = decode_fp8_block_scaled_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales.view(torch.int16), scales.view(torch.int16))

    actual = dequantize_fp8_block_scaled(payload, shape)
    assert actual[0, 0].item() == 1.0
    assert actual[0, 128].item() == 4.0
    assert actual[128, 0].item() == 8.0
    assert actual[128, 128].item() == 1.0


def test_block_scale_fp8_rejects_nonfinite_words_and_invalid_zero_tiles():
    codes = torch.zeros((2, 2), dtype=torch.uint8)
    scales = _bf16_words(0x3F80).reshape(1, 1)

    invalid_codes = codes.clone()
    invalid_codes[0, 0] = 0x7F
    with pytest.raises(ValueError, match="finite E4M3FN"):
        encode_fp8_block_scaled(invalid_codes, scales, (2, 2))

    with pytest.raises(ValueError, match="nonnegative finite BF16"):
        encode_fp8_block_scaled(codes, _bf16_words(0x8000).reshape(1, 1), (2, 2))

    nonzero_codes = codes.clone()
    nonzero_codes[0, 0] = 0x38
    with pytest.raises(ValueError, match="zero block scale"):
        encode_fp8_block_scaled(
            nonzero_codes, _bf16_words(0x0000).reshape(1, 1), (2, 2)
        )


def test_marlin_fp8_block_layout_round_trip_preserves_code_and_scale_words():
    shape = (256, 256)
    geometry = marlin_fp8_block_geometry("FP8_E4M3FN_BLOCK128_BF16S", shape)
    assert (geometry.n_tiles, geometry.k_tiles, geometry.groups) == (8, 8, 2)
    codes = (torch.arange(shape[0] * shape[1], dtype=torch.int32).reshape(shape) % 127).to(torch.uint8)
    scales = _bf16_words(0x3F80, 0x4000, 0x4040, 0x4080).reshape(2, 2)
    payload = encode_marlin_fp8_block(codes, scales, shape)

    assert len(payload) == encoded_size(
        "marlin-fp8-block128-v1", "FP8_E4M3FN_BLOCK128_BF16S", shape
    )
    decoded_codes, decoded_scales = decode_marlin_fp8_block_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales.view(torch.int16), scales.view(torch.int16))