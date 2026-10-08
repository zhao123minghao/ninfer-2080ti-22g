"""Convert the registered Qwen3.8-27B block-FP8 checkpoint to .ninfer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from tools.artifact.layouts import (
    decode_marlin_fp8_block_words,
    encode_fp8_block_scaled,
    encode_marlin_fp8_block,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import draft_head
from tools.convert.qwen3_8_27b import convert as base_convert

from . import inventory_fp8_block as inventory
from . import recipe_fp8_block as recipe


RECIPE_ID = "qwen3_8_27b_fp8-block128-v1"


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: family_conversion.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan
    inventory_specs: tuple[inventory.StoredObjectSpec, ...]
    marlin_layout: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_index(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    value = family_conversion.load_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map must be a nonempty object")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(shard, str)
        or not shard
        for name, shard in weight_map.items()
    ):
        raise ValueError(f"{index_path}: invalid weight_map entry")
    referenced = set(weight_map.values())
    actual = {path.name for path in model_dir.glob("*.safetensors")}
    if actual != referenced:
        raise ValueError(f"{model_dir}: safetensors shard set does not match the index")
    for shard in sorted(referenced):
        path = model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{path}: indexed shard is missing or empty")


def _validate_config(config: Mapping[str, object]) -> dict[str, object]:
    summary = base_convert.qwen3_6_convert.validate_config(config)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        raise ValueError("FP8 source is missing quantization_config")
    if (
        quantization.get("quant_method") != "fp8"
        or quantization.get("fmt") != "e4m3"
        or quantization.get("activation_scheme") != "dynamic"
        or tuple(quantization.get("weight_block_size", ())) != (128, 128)
    ):
        raise ValueError(
            "FP8 source must use dynamic E4M3 weights with 128x128 block scales"
        )
    return summary


def preflight_inventory() -> None:
    recipe.validate_recipe()


def selected_inventory_specs(marlin_layout: bool) -> tuple[inventory.StoredObjectSpec, ...]:
    if not marlin_layout:
        return inventory.OBJECT_SPECS
    return tuple(
        inventory.TensorSpec(spec.name, spec.shape, spec.format, inventory.MARLIN_BLOCK_SCALE_LAYOUT)
        if isinstance(spec, inventory.TensorSpec) and spec.format == inventory.FP8_BLOCK
        else spec
        for spec in inventory.OBJECT_SPECS
    )


def build_object_plan(
    resources: Mapping[str, bytes], inventory_specs: Sequence[inventory.StoredObjectSpec]
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory_specs, resources)


def preflight_conversion(model_dir: str | Path, *, marlin_layout: bool = False) -> ConversionPreflight:
    model = Path(model_dir)
    _validate_index(model)
    config = family_conversion.load_json(model / "config.json")
    config_summary = _validate_config(config)
    preflight_inventory()
    with ShardReader(model) as reader:
        source = recipe.preflight_source_reader(reader)
    resources = tuple(base_convert.load_resources(model))
    resource_map = {item.name: item.data for item in resources}
    inventory_specs = selected_inventory_specs(marlin_layout)
    object_plan = build_object_plan(resource_map, inventory_specs)
    ranking_path = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking_path, model)
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
        inventory_specs=inventory_specs,
        marlin_layout=marlin_layout,
    )


def _materialize_direct(
    object_name: str,
    reader: ShardReader,
    inventory_specs: Sequence[inventory.StoredObjectSpec],
) -> torch.Tensor:
    tensor = recipe.materialize_quantized_direct(object_name, reader)
    expected_shape = next(
        spec.shape for spec in inventory_specs
        if isinstance(spec, inventory.TensorSpec) and spec.name == object_name
    )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{object_name}: materialized shape {tuple(tensor.shape)} != {expected_shape}"
        )
    return tensor


def _materialize_official(
    object_name: str,
    reader: ShardReader,
    derived: Mapping[str, torch.Tensor],
    inventory_specs: Sequence[inventory.StoredObjectSpec],
) -> torch.Tensor:
    tensor = recipe.materialize_official(object_name, reader, dict(derived))
    expected_shape = next(
        spec.shape for spec in inventory_specs
        if isinstance(spec, inventory.TensorSpec) and spec.name == object_name
    )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{object_name}: materialized shape {tuple(tensor.shape)} != {expected_shape}"
        )
    return tensor


def build_conversion_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    weights_id = inventory.MARLIN_WEIGHTS_ID if preflight.marlin_layout else inventory.WEIGHTS_ID
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, weights_id),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.model_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["fp8_block_layout"] = {
        "format": inventory.FP8_BLOCK,
        "layout": (inventory.MARLIN_BLOCK_SCALE_LAYOUT
                   if preflight.marlin_layout else inventory.BLOCK_SCALE_LAYOUT),
        "block_shape": [128, 128],
        "scale_dtype": "BF16 multiplier",
        "source_matrices": len(recipe.FP8_BLOCK_SOURCES),
    }
    return report


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    marlin_layout: bool = False,
) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model, marlin_layout=marlin_layout)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.FP8_BLOCK_SOURCES)} block-FP8 matrices, "
        f"{preflight.source.source_tensor_count} source tensors, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {item.name: item.data for item in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    with ShardReader(model) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(
                inventory.MODEL_ID,
                inventory.MARLIN_WEIGHTS_ID if marlin_layout else inventory.WEIGHTS_ID,
            ),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(preflight.inventory_specs, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name in recipe.FP8_BLOCK_WEIGHTS_BY_NAME:
                    selected = recipe.FP8_BLOCK_WEIGHTS_BY_NAME[spec.name]
                    codes, scales = recipe.materialize_fp8_block_weight(selected, reader)
                    payload = (encode_marlin_fp8_block(codes, scales, spec.shape)
                               if preflight.marlin_layout
                               else encode_fp8_block_scaled(codes, scales, spec.shape))
                    if preflight.marlin_layout:
                        decoded_codes, decoded_scales = decode_marlin_fp8_block_words(
                            payload, spec.shape
                        )
                        if not torch.equal(decoded_codes, codes) or not torch.equal(
                            decoded_scales.view(torch.int16), scales.view(torch.int16)
                        ):
                            raise RuntimeError(f"{spec.name}: Marlin layout did not preserve FP8 words")
                elif spec.name in recipe.QUANTIZED_DIRECT_BY_NAME:
                    tensor = _materialize_direct(spec.name, reader, preflight.inventory_specs)
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                else:
                    tensor = _materialize_official(
                        spec.name, reader, derived, preflight.inventory_specs
                    )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(preflight.inventory_specs)}] {spec.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(model_dir),
        "out": str(out_path),
        "device": requested_device,
        "marlin_layout": marlin_layout,
    }
    report = build_conversion_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--marlin-layout", action="store_true",
                        help="persist block-FP8 tensors in Marlin's code/scale layout")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, marlin_layout=args.marlin_layout)


if __name__ == "__main__":
    main()