#!/usr/bin/env python3
"""Concurrency sweep against a local vLLM OpenAI-compatible server.

Measures end-to-end and steady-state aggregate throughput for a fixed prompt
length, with the number of in-flight requests set explicitly.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen3.8-27B-FP8"

SRC = """A local inference engine keeps one resident model and a bounded set of
in-flight requests. Each decode round is formed over the requests that are ready
at that moment, so the width of a round is the number of active sequences times
the width of the speculative window. The scheduler admits a request only when a
sequence slot and the key-value pages for its prompt are both available, and it
never preempts an admitted request. Pages are allocated in fixed-size blocks, so
the pool's capacity is a page count rather than a byte count, and a single long
request can therefore sterilize a large fraction of the pool even when the
average occupancy is low. Operators must distinguish the configured ceiling from
the occupied length: a service started with a large maximum context is not a
service that has that many tokens resident. Backends that page their cache
recompute nothing when a sequence moves between blocks, which is what makes the
block table indirection affordable on hardware without a hardware page walker.
Attention kernels read the block table for every key tile they walk, so the
indirection cost scales with the number of tiles rather than with the number of
blocks, and a shared page size trades tail waste against table size. Decode
throughput is reported per committed token, never per drafted token, because a
rejected draft costs a forward pass and returns nothing. Acceptance is a
trajectory quantity: two configurations with identical committed output can
still report different acceptance, because the draft and verify steps interleave
differently once a single near-tie token flips. For that reason a comparison of
speculative throughput across two configurations is only meaningful when the
prompt, the sampling parameters and the output budget are all held fixed, and
even then the counters should be reported alongside the rate. Prefill is a
separate regime from decode: it is compute bound, it amortizes the collective
costs across a full chunk, and it is where quantization schemes with a grouped
scale layout pay for their dequantization unless the weights are materialized in
a wide format first. The most useful single number for a serving deployment is
the aggregate committed token rate at the concurrency the deployment actually
sees, measured with a prompt length that resembles the real workload, because
that is the only figure that combines the scheduler, the attention kernels, the
weight streaming path and the speculative acceptance into one observable. A
microbenchmark can support an operator level claim but it cannot establish an
end-to-end improvement, and a kernel profile can explain a regression but it
cannot certify a quality claim. Every published rate should therefore carry the
hardware, the artifact identity, the effective quantization, the key-value
precision, the prompt length, the output budget and the concurrency it was
measured at, and it should be reproducible from the same command line with one
change at a time. When a configuration is compared against another engine, the
two must be given byte-identical inputs and the same sampling parameters, and
the comparison must state which differences remain uncontrolled, such as a
different quantization of the same fine-tune, a different key-value dtype, or a
different prefill chunk size that was never swept. The headline number alone is
not evidence; the conditions are part of the result."""


def _post(path, payload, timeout=7200):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(text):
    with _post("/tokenize", {"model": MODEL, "prompt": text}) as resp:
        return json.load(resp)["count"]


def build_prompt(target_tokens):
    unit = tokenize(SRC)
    reps = max(1, round(target_tokens / unit))
    text = "\n\n".join([SRC] * reps)
    return text, tokenize(text), reps, unit


def request(text, max_tokens, sink, nonce=None):
    start = time.perf_counter()
    ttft = None
    last = start
    completion = 0
    prompt_tokens = None
    content = text if nonce is None else "[session " + nonce + "]\n" + text
    try:
        with _post(
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                obj = json.loads(body)
                usage = obj.get("usage")
                if usage:
                    completion = usage.get("completion_tokens", completion)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - start
                        last = now
    except Exception as exc:  # noqa: BLE001
        sink.append({"error": repr(exc)})
        return
    end = time.perf_counter()
    sink.append(
        {
            "ttft": ttft,
            "total": end - start,
            "end": end,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
        }
    )


def run_point(concurrency, text, max_tokens):
    sink = []
    barrier = threading.Barrier(concurrency + 1)

    def worker(index):
        barrier.wait()
        request(text, max_tokens, sink, uuid.uuid4().hex[:12] + "-" + str(index))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    t0 = time.perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - t0

    errors = [r for r in sink if "error" in r]
    ok = [r for r in sink if "error" not in r]
    summary = {
        "concurrency": concurrency,
        "wall_s": round(wall, 3),
        "requests_ok": len(ok),
        "errors": errors,
    }
    if not ok:
        return summary
    completion = sum(r["completion_tokens"] for r in ok)
    prompt_total = sum(r["prompt_tokens"] or 0 for r in ok)
    max_ttft = max(r["ttft"] for r in ok)
    max_end = max(r["end"] for r in ok) - t0
    steady = max_end - max_ttft
    summary.update(
        {
            "prompt_tokens_each": ok[0]["prompt_tokens"],
            "completion_tokens_each": ok[0]["completion_tokens"],
            "ttft_s": [round(r["ttft"], 3) for r in ok],
            "total_s": [round(r["total"], 3) for r in ok],
            "ttft_mean_s": round(statistics.mean(r["ttft"] for r in ok), 3),
            "per_req_decode_tok_s": [
                round((r["completion_tokens"] - 1) / (r["total"] - r["ttft"]), 1)
                for r in ok
            ],
            "agg_output_tok_s_e2e": round(completion / wall, 1),
            "agg_output_tok_s_steady": round(completion / steady, 1) if steady > 0 else None,
            "agg_total_tok_s_e2e": round((completion + prompt_total) / wall, 1),
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, action="append", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--label", default="point")
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    text, actual, reps, unit = build_prompt(args.prompt_tokens)
    print(
        f"# corpus unit={unit} tok, repeats={reps}, raw prompt={actual} tok, "
        f"target={args.prompt_tokens}, max_tokens={args.max_tokens}",
        flush=True,
    )

    if not args.skip_warmup:
        warm = run_point(1, text, 16)
        print(f"# warmup: {json.dumps(warm)}", flush=True)

    for concurrency in args.concurrency:
        point = run_point(concurrency, text, args.max_tokens)
        point["label"] = args.label
        print("RESULT " + json.dumps(point), flush=True)


if __name__ == "__main__":
    sys.exit(main())
