from __future__ import annotations

import torch

from tools.artifact.layouts import (
    decode_fp8_block_scaled_words,
    decode_marlin_fp8_block_words,
    encode_fp8_block_scaled,
    encode_marlin_fp8_block,
)
from tools.convert.qwen3_8_27b import convert_fp8_block
from tools.convert.qwen3_8_27b import recipe_fp8_block


class _TensorReader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def test_fused_rows_reorder_codes_and_scale_tiles_together() -> None:
    raw_codes = torch.empty((512, 256), dtype=torch.uint8)
    for row_tile in range(4):
        for column_tile in range(2):
            raw_codes[
                row_tile * 128 : (row_tile + 1) * 128,
                column_tile * 128 : (column_tile + 1) * 128,
            ] = 0x38 + row_tile * 2 + column_tile
    source_codes = raw_codes.view(torch.float8_e4m3fn)
    source_scales = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.bfloat16
    )
    source = recipe_fp8_block.MatrixSource("source", (512, 256))
    weight = recipe_fp8_block.Fp8BlockWeightRecipe(
        "fused",
        (512, 256),
        (
            recipe_fp8_block.MatrixPart(source, (recipe_fp8_block.RowRange(256, 512),)),
            recipe_fp8_block.MatrixPart(source, (recipe_fp8_block.RowRange(0, 256),)),
        ),
    )
    reader = _TensorReader(
        {
            source.field("weight"): source_codes,
            source.field("weight_scale_inv"): source_scales,
        }
    )

    codes, scales = recipe_fp8_block.materialize_fp8_block_weight(weight, reader)
    expected_codes = torch.cat((raw_codes[256:], raw_codes[:256]), dim=0)
    expected_scales = torch.cat((source_scales[2:], source_scales[:2]), dim=0)
    assert torch.equal(codes, expected_codes)
    assert torch.equal(scales, expected_scales)

    payload = encode_fp8_block_scaled(codes, scales, weight.shape)
    decoded_codes, decoded_scales = decode_fp8_block_scaled_words(payload, weight.shape)
    assert torch.equal(decoded_codes, expected_codes)
    assert torch.equal(decoded_scales, expected_scales)


def test_marlin_inventory_relayouts_only_block_fp8_tensors() -> None:
    selected = convert_fp8_block.selected_inventory_specs(True)
    assert tuple(spec.name for spec in selected) == tuple(
        spec.name for spec in convert_fp8_block.inventory.OBJECT_SPECS
    )
    for original, marlin in zip(convert_fp8_block.inventory.OBJECT_SPECS, selected, strict=True):
        if isinstance(original, convert_fp8_block.inventory.TensorSpec) and (
            original.format == convert_fp8_block.inventory.FP8_BLOCK
        ):
            assert marlin.layout == convert_fp8_block.inventory.MARLIN_BLOCK_SCALE_LAYOUT
        else:
            assert marlin == original


def test_marlin_encoder_preserves_code_and_scale_words() -> None:
    codes = (torch.arange(512 * 256, dtype=torch.int32).reshape(512, 256) % 127).to(torch.uint8)
    scales = torch.tensor(range(1, 9), dtype=torch.int16).view(torch.bfloat16).reshape(4, 2)
    payload = encode_marlin_fp8_block(codes, scales, (512, 256))
    actual_codes, actual_scales = decode_marlin_fp8_block_words(payload, (512, 256))
    assert torch.equal(actual_codes, codes)
    assert torch.equal(actual_scales.view(torch.int16), scales.view(torch.int16))