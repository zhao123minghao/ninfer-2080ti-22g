#!/usr/bin/env python3
"""Export a source-Qwen3.8 FP8 greedy trajectory and its teacher-forced top-k logprobs.

Companion to `dump_source_logits.py`, which probes a single fixed context. This tool answers the
question that a single position cannot: where do source and Engine first disagree? It free-runs
G greedy tokens from one prompt, then re-evaluates every prefix of that trajectory (the whole
prompt, then prompt + generated[0..i) for each i) as a fresh single-token request, exporting the
next-token top-k at each position.

Both stages run inside one vLLM load, because a load costs minutes on this checkpoint and the
probing stage would otherwise pay it once per position. Logprob differences equal logit
differences at one position, so the exported top IDs and margins can be compared directly with
`tools/parity/qwen3_8_27b/engine_logits_probe.cpp`'s report over the same contexts.

The trajectory is what makes the two sides comparable at all: a free-running Engine and a
free-running source diverge permanently after the first different sample, so positions past that
point are different computations. Re-evaluating common prefixes is the standard teacher-forcing
construction that keeps both sides on the same context.

usage:
  python3 -m tools.parity.qwen3_8_27b.dump_source_trajectory \
    --model /path/to/Qwen3.8-27B-FP8 --ids-file prompt.ids --out report.json \
    [--tokens 16] [--top-k 8] [--devices 0,1] [--gpu-memory-utilization 0.9]
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
    result.add_argument("--tokens", type=int, default=16, help="greedy tokens to free-run first")
    result.add_argument("--top-k", type=int, default=8)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return result


def top_entries(candidates: dict) -> list[dict]:
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
    return entries


def main() -> None:
    args = parser().parse_args()
    if not args.model.is_dir():
        raise ValueError("--model must name a source checkpoint directory")
    if args.tokens < 1:
        raise ValueError("--tokens must be positive")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    prompt_ids = read_ids(args.ids_file)
    devices = parse_devices(args.devices)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(args.model),
        tensor_parallel_size=len(devices),
        max_model_len=len(prompt_ids) + args.tokens + 1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    greedy = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.tokens, ignore_eos=True, detokenize=False
    )
    free_run = engine.generate([{"prompt_token_ids": prompt_ids}], greedy, use_tqdm=False)[0]
    trajectory = [int(token) for token in free_run.outputs[0].token_ids]
    if len(trajectory) != args.tokens:
        raise RuntimeError(
            f"free run produced {len(trajectory)} tokens, expected {args.tokens}"
        )

    contexts = [prompt_ids + trajectory[:index] for index in range(args.tokens + 1)]
    probe = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True, logprobs=args.top_k,
        detokenize=False,
    )
    probed = engine.generate(
        [{"prompt_token_ids": context} for context in contexts], probe, use_tqdm=False
    )
    if len(probed) != len(contexts):
        raise RuntimeError("vLLM returned the wrong number of probe results")

    probes = []
    for index, output in enumerate(probed):
        result = output.outputs[0]
        if len(result.token_ids) != 1 or len(result.logprobs) != 1:
            raise RuntimeError("vLLM did not return exactly one next-token distribution")
        entries = top_entries(result.logprobs[0])
        margin = entries[0]["logprob"] - entries[1]["logprob"] if len(entries) >= 2 else 0.0
        probes.append(
            {
                "position": len(contexts[index]),
                "context_tokens": len(contexts[index]),
                "sampled_token_id": int(result.token_ids[0]),
                "argmax_token_id": entries[0]["token_id"],
                "margin": float(margin),
                "top": entries,
            }
        )

    report = {
        "format": "ninfer_source_trajectory_v1",
        "source_model": str(args.model.resolve()),
        "input_ids_file": str(args.ids_file.resolve()),
        "input_tokens": len(prompt_ids),
        "cuda_visible_devices": ",".join(devices),
        "tensor_parallel_size": len(devices),
        "quantization": "checkpoint-config:fp8",
        "greedy_tokens": args.tokens,
        "trajectory": trajectory,
        "probes": probes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"source trajectory: {args.tokens} tokens, {len(probes)} probes -> {args.out}")
    print("  trajectory:", trajectory)
    for entry in probes:
        print(
            f"  position {entry['position']}: argmax {entry['argmax_token_id']} "
            f"(sampled {entry['sampled_token_id']}, margin {entry['margin']:.4f})"
        )


if __name__ == "__main__":
    main()
