#!/usr/bin/env python3
"""Export source-Qwen3.8 FP8 next-token top-k logprobs for fixed token IDs.

The report is a source-aligned quality oracle for the block-FP8 .ninfer artifact.
Logprob differences equal logit differences at one position, so top IDs and margins
can be compared directly with the Engine's captured logits.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_ids(path: Path) -> list[int]:
    try:
        values = [int(value) for value in path.read_text(encoding="utf-8").split()]
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not values:
        raise ValueError("token-ID input is empty")
    if any(value < 0 for value in values):
        raise ValueError("token IDs must be non-negative")
    return values


def parse_devices(text: str) -> tuple[str, ...]:
    devices = tuple(value.strip() for value in text.split(",") if value.strip())
    if len(devices) < 1 or any(not value.isdigit() for value in devices):
        raise ValueError("--devices must be a nonempty comma-separated CUDA device list")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not repeat a CUDA device")
    return devices


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True, type=Path)
    result.add_argument("--ids-file", required=True, type=Path)
    result.add_argument("--out", required=True, type=Path)
    result.add_argument("--devices", default="0,1")
    result.add_argument("--top-k", type=int, default=8)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return result


def main() -> None:
    args = parser().parse_args()
    if not args.model.is_dir():
        raise ValueError("--model must name a source checkpoint directory")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    token_ids = read_ids(args.ids_file)
    devices = parse_devices(args.devices)

    # vLLM reads CUDA_VISIBLE_DEVICES while it initializes its worker processes.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(args.model),
        tensor_parallel_size=len(devices),
        max_model_len=len(token_ids) + 1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        ignore_eos=True,
        logprobs=args.top_k,
        detokenize=False,
    )
    result = engine.generate(
        [{"prompt_token_ids": token_ids}], params, use_tqdm=False
    )[0].outputs[0]
    if len(result.token_ids) != 1 or len(result.logprobs) != 1:
        raise RuntimeError("vLLM did not return exactly one next-token logprob distribution")
    candidates = result.logprobs[0]
    entries = sorted(
        (
            {
                "token_id": int(token_id),
                "logprob": float(value.logprob),
                "rank": int(value.rank),
            }
            for token_id, value in candidates.items()
        ),
        key=lambda value: value["logprob"],
        reverse=True,
    )
    if not entries:
        raise RuntimeError("vLLM returned an empty next-token distribution")
    report = {
        "format": "ninfer_source_top_logprobs_v1",
        "source_model": str(args.model.resolve()),
        "input_ids_file": str(args.ids_file.resolve()),
        "input_tokens": len(token_ids),
        "cuda_visible_devices": ",".join(devices),
        "tensor_parallel_size": len(devices),
        "quantization": "checkpoint-config:fp8",
        "generated_token_id": int(result.token_ids[0]),
        "top_logprobs": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()