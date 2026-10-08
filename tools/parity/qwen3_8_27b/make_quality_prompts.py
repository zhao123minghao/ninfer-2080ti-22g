#!/usr/bin/env python3
"""Generate token-ID prompt files for the source-aligned quality gate.

Companion to `dump_source_trajectory.py` / `engine_logits_probe.cpp`. This tool turns three
representative chat workloads (short text, code, reasoning) into the same token-ID files the two
probe tools consume, using the exact checkpoint tokenizer + chat template, so the three new
quality probes share the same prompt construction as the existing AIME26-30 probe.

The workload content is fixed here (three distinct generation regimes), not read from user
input, so the quality corpus is reproducible and provenance is explicit in the output.

usage:
  python3 -m tools.parity.qwen3_8_27b.make_quality_prompts \
    --model /path/to/Qwen3.8-27B-FP8 --out-dir .scratch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM = (
    "Think carefully to solve the problem. Return only the final answer, "
    "with no explanation or intermediate work."
)

WORKLOADS: dict[str, dict[str, str]] = {
    "shorttext": {
        "role": "user",
        "content": (
            "Write a short paragraph about the benefits of regular exercise for cardiovascular "
            "health, written for a general audience."
        ),
    },
    "code": {
        "role": "user",
        "content": (
            "Write a Python function that returns the longest palindromic substring of a given "
            "string, handling empty input, and include two assert statements as test cases."
        ),
    },
    "reasoning": {
        "role": "user",
        "content": (
            "A farmer has chickens and rabbits in a pen. Together the animals have 35 heads and "
            "94 legs. How many chickens and how many rabbits are there? Give only the two numbers."
        ),
    },
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True, type=Path)
    result.add_argument("--out-dir", required=True, type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if not args.model.is_dir():
        raise ValueError("--model must name a source checkpoint directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    manifest = {}
    for name, user_message in WORKLOADS.items():
        messages = [{"role": "system", "content": SYSTEM}, user_message]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        ids = [int(value) for value in encoded.input_ids]
        path = args.out_dir / f"quality_{name}.prompt.ids"
        path.write_text(" ".join(str(value) for value in ids) + "\n", encoding="utf-8")
        manifest[name] = {"ids_file": str(path.resolve()), "tokens": len(ids)}
        print(f"{name}: {len(ids)} tokens -> {path}")

    manifest_path = args.out_dir / "quality_prompts.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
