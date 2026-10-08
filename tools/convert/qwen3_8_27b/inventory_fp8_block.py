"""Persistent-object contract for Qwen3.8-27B block-scaled FP8 weights."""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FP32,
    I32,
    Q4,
    Q5,
    Q6,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    tensor_spec,
)

from . import inventory_nvfp4 as nvfp4


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "fp8-block128"
MARLIN_WEIGHTS_ID = "fp8-block128-marlin"
TARGET_KEY = "qwen3_8_27b"

FP8_BLOCK = "FP8_E4M3FN_BLOCK128_BF16S"
BLOCK_SCALE_LAYOUT = "blockscale-m128-k128-v1"
MARLIN_BLOCK_SCALE_LAYOUT = "marlin-fp8-block128-v1"
FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, FP8_BLOCK)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT, BLOCK_SCALE_LAYOUT)

RESOURCE_SPECS = nvfp4.RESOURCE_SPECS
FULL_ATTENTION_LAYERS = nvfp4.FULL_ATTENTION_LAYERS
GDN_LAYERS = nvfp4.GDN_LAYERS

_MTP_BLOCK_WEIGHTS = frozenset(
    (
        "mtp/layer/attention/query_key_gate_value",
        "mtp/layer/attention/output",
        "mtp/layer/mlp/gate_up",
        "mtp/layer/mlp/down",
    )
)
_BF16_WEIGHTS = frozenset(
    ("text/token_embedding", "text/output_head", "mtp/input_projection")
)


def _convert_spec(spec: TensorSpec) -> TensorSpec:
    if spec.name in _BF16_WEIGHTS:
        return tensor_spec(spec.name, spec.shape, BF16)
    if spec.format in (nvfp4.NVFP4, nvfp4.FP8) or spec.name in _MTP_BLOCK_WEIGHTS:
        return TensorSpec(spec.name, spec.shape, FP8_BLOCK, BLOCK_SCALE_LAYOUT)
    return spec


TEXT_CORE_TENSOR_SPECS = tuple(
    _convert_spec(spec)
    for spec in nvfp4.TEXT_CORE_TENSOR_SPECS
    if not spec.name.endswith("/input_scale_divisor")
)
DRAFT_HEAD_TENSOR_SPECS = tuple(
    _convert_spec(spec) for spec in nvfp4.DRAFT_HEAD_TENSOR_SPECS
)
MTP_TENSOR_SPECS = tuple(_convert_spec(spec) for spec in nvfp4.MTP_TENSOR_SPECS)
VISION_TENSOR_SPECS = nvfp4.VISION_TENSOR_SPECS
TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = nvfp4.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = nvfp4.ALIAS_SPECS

__all__ = [
    "ALIAS_SPECS",
    "BLOCK_SCALE_LAYOUT",
    "FP8_BLOCK",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "MARLIN_BLOCK_SCALE_LAYOUT",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "OBJECT_SPECS",
    "RESOURCE_SPECS",
    "TENSOR_SPECS",
    "TARGET_KEY",
    "WEIGHTS_ID",
]