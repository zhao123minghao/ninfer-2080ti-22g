"""Closed source recipe for the Qwen3.8-27B 128x128 block-FP8 artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import recipe as base_recipe

from . import inventory_fp8_block as inventory
from . import inventory_nvfp4 as nvfp4_inventory
from . import recipe_nvfp4 as nvfp4_recipe


@dataclass(frozen=True, slots=True)
class RowRange:
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class MatrixSource:
    name: str
    shape: tuple[int, int]

    def field(self, suffix: str) -> str:
        return f"{self.name}.{suffix}"


@dataclass(frozen=True, slots=True)
class MatrixPart:
    source: MatrixSource
    rows: tuple[RowRange, ...]

    @property
    def output_rows(self) -> int:
        return sum(item.rows for item in self.rows)


@dataclass(frozen=True, slots=True)
class Fp8BlockWeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]


def _source(name: str, n: int, k: int) -> MatrixSource:
    return MatrixSource(name, (n, k))


def _all(source: MatrixSource) -> MatrixPart:
    return MatrixPart(source, (RowRange(0, source.shape[0]),))


def _q_part(source: MatrixSource, gate: bool) -> MatrixPart:
    begin = 256 if gate else 0
    return MatrixPart(
        source,
        tuple(
            RowRange(head * 512 + begin, head * 512 + begin + 256)
            for head in range(24)
        ),
    )


def _mtp_weight_recipes() -> tuple[Fp8BlockWeightRecipe, ...]:
    prefix = "mtp.layers.0."
    query = _source(prefix + "self_attn.q_proj", 12288, 5120)
    key = _source(prefix + "self_attn.k_proj", 1024, 5120)
    value = _source(prefix + "self_attn.v_proj", 1024, 5120)
    output = _source(prefix + "self_attn.o_proj", 5120, 6144)
    gate = _source(prefix + "mlp.gate_proj", 17408, 5120)
    up = _source(prefix + "mlp.up_proj", 17408, 5120)
    down = _source(prefix + "mlp.down_proj", 5120, 17408)
    return (
        Fp8BlockWeightRecipe(
            "mtp/layer/attention/query_key_gate_value",
            (14336, 5120),
            (_q_part(query, False), _all(key), _q_part(query, True), _all(value)),
        ),
        Fp8BlockWeightRecipe(
            "mtp/layer/attention/output", output.shape, (_all(output),)
        ),
        Fp8BlockWeightRecipe(
            "mtp/layer/mlp/gate_up",
            (34816, 5120),
            (_all(gate), _all(up)),
        ),
        Fp8BlockWeightRecipe("mtp/layer/mlp/down", down.shape, (_all(down),)),
    )


def _layer_weight_recipes() -> tuple[Fp8BlockWeightRecipe, ...]:
    recipes = nvfp4_recipe.FP8_WEIGHT_RECIPES + tuple(
        nvfp4_recipe.Fp8WeightRecipe(item.object_name, item.shape, item.parts)
        for item in nvfp4_recipe.NVFP4_WEIGHT_RECIPES
    )
    return tuple(
        Fp8BlockWeightRecipe(item.object_name, item.shape, item.parts)
        for item in recipes
        if item.object_name != "text/output_head"
    )


FP8_BLOCK_WEIGHT_RECIPES = _layer_weight_recipes() + _mtp_weight_recipes()
FP8_BLOCK_WEIGHTS_BY_NAME = {
    item.object_name: item for item in FP8_BLOCK_WEIGHT_RECIPES
}
FP8_BLOCK_SOURCES = tuple(
    dict.fromkeys(part.source for item in FP8_BLOCK_WEIGHT_RECIPES for part in item.parts)
)

MTP_BLOCK_OBJECTS = frozenset(
    item.object_name for item in _mtp_weight_recipes()
)
QUANTIZED_DIRECT_RECIPES = nvfp4_recipe.QUANTIZED_DIRECT_RECIPES
QUANTIZED_DIRECT_BY_NAME = nvfp4_recipe.QUANTIZED_DIRECT_BY_NAME
OFFICIAL_RECIPES = tuple(
    item
    for item in nvfp4_recipe.OFFICIAL_RECIPES
    if item.object_name not in MTP_BLOCK_OBJECTS
) + (base_recipe.RECIPES_BY_NAME["text/output_head"],)
OFFICIAL_RECIPES_BY_NAME = {item.object_name: item for item in OFFICIAL_RECIPES}


def _validate_matrix_recipe(item: Fp8BlockWeightRecipe) -> None:
    n, k = item.shape
    if n % 128 or k % 128 or not item.parts:
        raise ValueError(f"{item.object_name}: block-FP8 shape must be 128-aligned")
    if sum(part.output_rows for part in item.parts) != n:
        raise ValueError(f"{item.object_name}: fused row geometry does not match")
    for part in item.parts:
        if part.source.shape[1] != k or not part.rows:
            raise ValueError(f"{item.object_name}: incompatible source geometry")
        for rows in part.rows:
            if (
                rows.begin < 0
                or rows.end > part.source.shape[0]
                or rows.begin >= rows.end
                or rows.begin % 128
                or rows.end % 128
            ):
                raise ValueError(
                    f"{item.object_name}: source row slice breaks a 128-row scale tile"
                )


def validate_recipe() -> None:
    family_recipe.validate_recipe_coverage(
        QUANTIZED_DIRECT_RECIPES, nvfp4_recipe.QUANTIZED_DIRECT_SPECS
    )
    block_names = {item.name for item in inventory.TENSOR_SPECS if item.format == inventory.FP8_BLOCK}
    if (
        len(FP8_BLOCK_WEIGHT_RECIPES),
        len(FP8_BLOCK_WEIGHTS_BY_NAME),
        len(FP8_BLOCK_SOURCES),
        len(QUANTIZED_DIRECT_RECIPES),
    ) != (260, 260, 407, 401):
        raise ValueError("Qwen3.8 block-FP8 recipe inventory is incomplete")
    if set(FP8_BLOCK_WEIGHTS_BY_NAME) != block_names:
        raise ValueError("block-FP8 recipes do not match artifact tensor inventory")
    routed = (
        set(FP8_BLOCK_WEIGHTS_BY_NAME)
        | set(QUANTIZED_DIRECT_BY_NAME)
        | set(OFFICIAL_RECIPES_BY_NAME)
    )
    expected = {item.name for item in inventory.TENSOR_SPECS}
    if routed != expected:
        raise ValueError(
            "source routes do not cover the complete tensor inventory: "
            f"missing={sorted(expected - routed)}, extra={sorted(routed - expected)}"
        )
    for item in FP8_BLOCK_WEIGHT_RECIPES:
        _validate_matrix_recipe(item)


def _merge_requirement(
    result: dict[str, tuple[tuple[int, ...], str]],
    name: str,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    signature = (shape, dtype)
    previous = result.setdefault(name, signature)
    if previous != signature:
        raise ValueError(f"inconsistent source declaration for {name}")


def _source_requirements() -> dict[str, tuple[tuple[int, ...], str]]:
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for source in FP8_BLOCK_SOURCES:
        n, k = source.shape
        _merge_requirement(result, source.field("weight"), (n, k), "F8_E4M3")
        _merge_requirement(
            result,
            source.field("weight_scale_inv"),
            (n // 128, k // 128),
            "BF16",
        )
    for source in family_recipe.source_requirements(QUANTIZED_DIRECT_RECIPES).values():
        _merge_requirement(result, source.name, source.shape, source.dtype)
    for source in family_recipe.source_requirements(OFFICIAL_RECIPES).values():
        _merge_requirement(result, source.name, source.shape, source.dtype)
    return result


SOURCE_REQUIREMENTS = _source_requirements()


def preflight_source_reader(reader: ShardReader) -> family_recipe.SourcePreflight:
    expected = set(SOURCE_REQUIREMENTS)
    actual = set(reader.names)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"FP8 source is missing {sorted(missing)[0]}")
    if extra:
        raise ValueError(f"FP8 source has unclaimed tensor {sorted(extra)[0]}")

    metadata = reader.metadata(reader.names)
    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, (shape, dtype) in SOURCE_REQUIREMENTS.items():
        item = metadata[name]
        if item.shape != shape or item.dtype != dtype:
            raise ValueError(
                f"{name}: source signature {(item.shape, item.dtype)} != {(shape, dtype)}"
            )
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        shards.add(item.shard)
    return family_recipe.SourcePreflight(
        recipe_count=(
            len(FP8_BLOCK_WEIGHT_RECIPES)
            + len(QUANTIZED_DIRECT_RECIPES)
            + len(OFFICIAL_RECIPES)
        ),
        source_tensor_count=len(SOURCE_REQUIREMENTS),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def _select_rows(tensor: torch.Tensor, part: MatrixPart) -> torch.Tensor:
    pieces = [tensor.narrow(0, item.begin, item.rows) for item in part.rows]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def _select_scale_rows(tensor: torch.Tensor, part: MatrixPart) -> torch.Tensor:
    pieces = [
        tensor.narrow(0, item.begin // 128, item.rows // 128)
        for item in part.rows
    ]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def materialize_fp8_block_weight(
    item: Fp8BlockWeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor]:
    code_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    source_words: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor]] = {}
    for part in item.parts:
        words = source_words.get(part.source)
        if words is None:
            n, k = part.source.shape
            codes = reader.get(part.source.field("weight"))
            scales = reader.get(part.source.field("weight_scale_inv"))
            if (
                codes.dtype != torch.float8_e4m3fn
                or tuple(codes.shape) != (n, k)
                or scales.dtype != torch.bfloat16
                or tuple(scales.shape) != (n // 128, k // 128)
            ):
                raise ValueError(
                    f"{part.source.name}: materialized block-FP8 source signature mismatch"
                )
            words = (codes.view(torch.uint8), scales)
            source_words[part.source] = words
        code_parts.append(_select_rows(words[0], part))
        scale_parts.append(_select_scale_rows(words[1], part))

    codes = code_parts[0] if len(code_parts) == 1 else torch.cat(code_parts, dim=0)
    scales = scale_parts[0] if len(scale_parts) == 1 else torch.cat(scale_parts, dim=0)
    if tuple(codes.shape) != item.shape or tuple(scales.shape) != (
        item.shape[0] // 128,
        item.shape[1] // 128,
    ):
        raise ValueError(f"{item.object_name}: materialized block-FP8 shape mismatch")
    return codes.contiguous(), scales.contiguous()


def materialize_quantized_direct(
    object_name: str,
    reader: ShardReader,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        QUANTIZED_DIRECT_BY_NAME[object_name], reader
    )


def materialize_official(
    object_name: str,
    reader: ShardReader,
    derived_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        OFFICIAL_RECIPES_BY_NAME[object_name], reader, derived_tensors
    )


validate_recipe()


__all__ = [
    "FP8_BLOCK_SOURCES",
    "FP8_BLOCK_WEIGHT_RECIPES",
    "FP8_BLOCK_WEIGHTS_BY_NAME",
    "OFFICIAL_RECIPES_BY_NAME",
    "QUANTIZED_DIRECT_BY_NAME",
    "SOURCE_REQUIREMENTS",
    "materialize_fp8_block_weight",
    "materialize_official",
    "materialize_quantized_direct",
    "preflight_source_reader",
    "validate_recipe",
]