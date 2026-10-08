"""Attach Qwen3.8-27B DFlash2 weights to an existing groupwise-int artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Sequence

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
    ResourceObject,
    ResourceSpec,
    TensorObject,
    TensorSpec,
)

from . import inventory
from .convert_gguf import DirectObject, build_dflash2_objects


DFLASH_OBJECT_COUNT = 66
GROUPWISE_IDENTITY = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)


def _object_spec(obj: ArtifactObject) -> ResourceSpec | TensorSpec:
    if isinstance(obj, TensorObject):
        return TensorSpec(
            obj.name,
            obj.shape,
            obj.format,
            obj.layout,
            encoded_bytes=obj.bytes,
        )
    if isinstance(obj, ResourceObject):
        return ResourceSpec(obj.name, obj.encoding, obj.bytes)
    raise TypeError(f"unsupported artifact object: {type(obj).__name__}")


def _preflight(
    artifact_path: str | Path, out_path: str | Path
) -> tuple[Path, Path, Path, dict[str, int]]:
    source_path = Path(artifact_path).resolve(strict=True)
    output_path = Path(out_path).resolve()
    report_path = Path(str(output_path) + ".conversion.json")
    if source_path == output_path:
        raise ValueError("output must not replace the source artifact")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if report_path.exists():
        raise FileExistsError(f"conversion report already exists: {report_path}")

    with Artifact.open(source_path) as source:
        if source.identity != GROUPWISE_IDENTITY:
            raise ValueError(
                "expected qwen3.8-27b/groupwise-int artifact, got "
                f"{source.identity.model_id}/{source.identity.weights_id}"
            )
        if any(obj.name.startswith("dflash/") for obj in source.objects):
            raise ValueError("source artifact already contains DFlash objects")
        summary = {"objects": len(source.objects), "bytes": source.file_bytes}
    return source_path, output_path, report_path, summary


def append_dflash2_objects(
    artifact_path: str | Path,
    out_path: str | Path,
    dflash_objects: Sequence[DirectObject],
) -> Path:
    """Copy the base object's bytes unchanged and append validated DFlash2 tensors."""
    source_path, output_path, _, _ = _preflight(artifact_path, out_path)
    objects = tuple(dflash_objects)
    if not objects:
        raise ValueError("DFlash2 object inventory must not be empty")
    names: set[str] = set()
    for obj in objects:
        spec = obj.spec
        if not spec.name.startswith("dflash/"):
            raise ValueError(f"not a DFlash2 object: {spec.name}")
        if spec.name in names:
            raise ValueError(f"duplicate DFlash2 object: {spec.name}")
        names.add(spec.name)
        if spec.format != "BF16" or spec.layout != "contiguous-le-v1":
            raise ValueError(f"DFlash2 object must use contiguous BF16: {spec.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with Artifact.open(source_path) as source:
            specs = [_object_spec(obj) for obj in source.objects]
            specs.extend(obj.spec for obj in objects)
            with ArtifactWriter(temporary_path, source.identity, specs) as writer:
                if writer.objects[:len(source.objects)] != source.objects:
                    raise RuntimeError("copying would change the base object directory")
                for obj in source.objects:
                    payload = source.payload(obj)
                    try:
                        writer.write(obj.name, payload)
                    finally:
                        payload.release()
                for obj in objects:
                    writer.write(obj.spec.name, obj.data)
                planned_objects = writer.objects
            source_mode = stat.S_IMODE(source_path.stat().st_mode)

        with Artifact.open(temporary_path) as result:
            if result.identity != GROUPWISE_IDENTITY or result.objects != planned_objects:
                raise RuntimeError("written artifact failed structural verification")

        os.chmod(temporary_path, source_mode)
        os.link(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    source_path, output_path, report_path, base_summary = _preflight(
        args.artifact, args.out
    )
    dflash_objects = build_dflash2_objects(args.dflash2_model)
    if len(dflash_objects) != DFLASH_OBJECT_COUNT:
        raise RuntimeError(
            f"expected {DFLASH_OBJECT_COUNT} DFlash2 tensors, got {len(dflash_objects)}"
        )
    append_dflash2_objects(source_path, output_path, dflash_objects)

    report = {
        "operation": "append-qwen3.8-dflash2-v1",
        "identity": {
            "model_id": GROUPWISE_IDENTITY.model_id,
            "weights_id": GROUPWISE_IDENTITY.weights_id,
        },
        "base_artifact": {
            "path": str(source_path),
            "bytes": base_summary["bytes"],
            "objects": base_summary["objects"],
            "payloads": "copied byte-for-byte",
        },
        "dflash2_checkpoint": str(args.dflash2_model.resolve()),
        "dflash2": {
            "target_layer_ids": [5, 19, 33, 47, 61],
            "block_size": 8,
            "tensor_count": len(dflash_objects),
            "format": "BF16",
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "objects": base_summary["objects"] + len(dflash_objects),
        },
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: copied {base_summary['objects']} base objects and appended "
        f"{len(dflash_objects)} BF16 DFlash2 tensors; output={output_path}"
    )


if __name__ == "__main__":
    main()