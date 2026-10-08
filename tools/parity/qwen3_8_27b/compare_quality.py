#!/usr/bin/env python3
"""Compare a source trajectory report against an Engine logits report, position by position.

Both reports are produced over the same token-ID contexts (teacher-forced prefixes of one prompt +
greedy trajectory), so the next-token top-k at each position is the same computation on two sides:
`dump_source_trajectory.py` (source, logprob units) and `engine_logits_probe` (Engine, logit units).
Logprob differences equal logit differences at one position, so top IDs and margins compare directly.

usage:
  python3 -m tools.parity.qwen3_8_27b.compare_quality \
    --source report.source.json --engine report.engine.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", required=True, type=Path)
    result.add_argument("--engine", required=True, type=Path)
    return result


def top_set(probe: dict) -> set[int]:
    return {int(entry["token_id"]) for entry in probe["top"]}


def main() -> None:
    args = parser().parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    engine = json.loads(args.engine.read_text(encoding="utf-8"))

    source_probes = {int(p["position"]): p for p in source["probes"]}
    engine_probes = {int(p["position"]): p for p in engine["probes"]}
    positions = sorted(set(source_probes) & set(engine_probes))
    if not positions:
        raise SystemExit("no positions in common between source and engine reports")

    argmax_hits = 0
    overlap_sum = 0
    overlap_min = None
    margin_max_abs = 0.0
    margin_abs_sum = 0.0
    rows = []
    for position in positions:
        s, e = source_probes[position], engine_probes[position]
        s_top, e_top = top_set(s), top_set(e)
        overlap = len(s_top & e_top)
        overlap_sum += overlap
        overlap_min = overlap if overlap_min is None else min(overlap_min, overlap)
        margin_abs = abs(float(s["margin"]) - float(e["margin"]))
        margin_max_abs = max(margin_max_abs, margin_abs)
        margin_abs_sum += margin_abs
        hit = int(s["argmax_token_id"]) == int(e["argmax_token_id"])
        argmax_hits += int(hit)
        rows.append(
            {
                "position": position,
                "argmax_match": hit,
                "source_argmax": int(s["argmax_token_id"]),
                "engine_argmax": int(e["argmax_token_id"]),
                "top8_overlap": overlap,
                "source_margin": float(s["margin"]),
                "engine_margin": float(e["margin"]),
                "margin_abs_delta": margin_abs,
            }
        )

    total = len(positions)
    summary = {
        "positions": total,
        "argmax_agreement": argmax_hits / total,
        "top8_overlap_min": overlap_min,
        "top8_overlap_mean": overlap_sum / total,
        "margin_abs_delta_max": margin_max_abs,
        "margin_abs_delta_mean": margin_abs_sum / total,
    }
    print(json.dumps(summary, indent=2))
    for row in rows:
        if not row["argmax_match"] or row["top8_overlap"] < 8:
            print(
                f"  mismatch @ {row['position']}: argmax {row['source_argmax']} vs "
                f"{row['engine_argmax']}, overlap {row['top8_overlap']}/8, "
                f"margin {row['source_margin']:.4f} vs {row['engine_margin']:.4f}"
            )
    print(
        f"argmax {argmax_hits}/{total} = {summary['argmax_agreement']:.4f}, "
        f"top8 overlap {overlap_min}..{summary['top8_overlap_mean']:.3f}, "
        f"margin |Δ| max {margin_max_abs:.4f} mean {summary['margin_abs_delta_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
