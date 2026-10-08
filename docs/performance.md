# Serving performance

## V100X2 measurement and acceptance

### Prefill and external optimisation review

The promoted route is measured on the real Q4_K_M artifact with INT8 KV, TP2 and
180,000-token capacity. With `prefill_chunk=4096`, an 8,192-token prompt took **4.897 s**
to prefill (**1,672.9 tok/s**); the fixed 85,000-token code corpus took **67.597–67.943 s**
(**1,251.0–1,257.5 tok/s**, two runs). The latter has exactly 85,000 occupied prompt tokens.
These are prefill rates, not decode rates, and both exceed the requested 1,000 tok/s target.
The 4,096-token chunk is the V100X2 launcher default; the Engine and generic CLI retain their
target-agnostic 1,024-token default.

On the 8,192-token probe with this kernel, a chunk sweep measured **1,454.4**, **1,599.3**,
**1,636.1**, **1,672.9**, **1,672.3**, and **1,195.2 tok/s** for chunk sizes 1,024, 2,048,
3,072, 4,096, 5,120, and 8,192 respectively. The 85,000-token corpus measured **1,133.0**,
**1,213.4**, and **1,251.0 tok/s** at 1,024, 2,048, and 4,096 respectively. Larger chunks
increase the startup workspace reservation; at 4,096 it was **1.49 GiB per device**, still within
the two 16 GB cards' capacity at 180,000 context. Values are individual cold-prefill runs, not
averages across repeated campaigns.

Before the multi-output route was promoted, an 8,192-token probe took **27.130 s** with
`prefill_chunk=1024` and **27.075 s** with `prefill_chunk=4096`. A CUDA Nsight Systems capture
attributed **89.0%** of GPU kernel time to the existing Volta GGML-K Tensor-Core GEMM (`1036`
launches, `11.544 s` aggregate in the 2,048-token capture). Those measurements are retained as
historical attribution only; they are not the current implementation result.

Two controlled tile experiments were rejected: using the 32-token GEMM tile for long prefill
increased the 8K probe to **30.828 s**, and a 128-token/512-thread tile increased it to
**38.725 s**. Both were reverted. The independent GGML-K FP64-oracle suite remains passing.
The promoted SM70 route decodes each Q4_K block with one cooperative block-wide pass,
materializes GGML-K rows into caller-owned FP16 workspace, and uses the existing CUTLASS Volta
Tensor-Core GEMM, including an FP32 output path for GDN control projection. The packed Q4_K/Q6_K
bytes remain unchanged. The first two generated IDs were unchanged (`71093, 10504`) in the
controlled route comparison. These are prefill results only;
the published decode acceptance remains the separately measured 53.4075 tok/s until a new long
decode campaign is run.

The external references and integration boundaries are:

- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), an SM70-focused vLLM fork. Its
  Flash-V100 attention and quantized kernels are not drop-in compatible with NInfer's preserved
  GGUF Q4_K/Q6_K storage; porting one requires a separate oracle and graph-capture qualification.
- [DFlash2](https://inco.ai/blog/dflash2/) and the [Qwen3.8-27B drafter](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
  now integrated as an optional five-layer BF16 draft route with dynamic grouped convolution,
  lattice selection, and TP2 replicated verification. A short two-V100 smoke run completed eight generated tokens with 100% acceptance; no long-context
  throughput claim is published yet. A temporary llama.cpp `sm_70` build was used only as an external
  reference: with the same local Q4_K_M target, Q8 KV, an approximately 6,020-token prompt and
  128 generated tokens, it measured 34.2 tok/s without speculation, 65.6 tok/s with DFlash2
  draft=3, and 85.0 tok/s with DFlash2 draft=7 on V100×2. These are not NInfer measurements.
- [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp), which stores completed KV
  blocks in host RAM and retrieves a query-selected subset into a bounded GPU window. That is an
  approximate attention policy, not a transparent full-180K KV spill: it changes which history
  participates in attention. The V100X2 contract therefore keeps complete-context semantics and
  does not silently substitute KVMem retrieval.

### PCIe-only transport optimization

The two V100-SXM2 cards are behind a translated `DMA-FQ` IOMMU domain and have no active NVLink.
That topology makes transport, rather than Q4/Q6 arithmetic, the common ceiling: TP2 executes 128
hidden-width all-reduces per committed token (64 layers × mixer/MLP row-parallel outputs), each
carrying 10 KiB at batch one. Quantization therefore cannot remove this fixed schedule, explaining
why Q1, Q4, and Q6 runs converge near the same peak.

The runtime keeps direct UVA P2P when startup qualifies it. On this host it selects a graph-safe
explicit fallback: one pinned host slot per rank, asynchronous D2H exports followed by peer H2D
imports, with event ordering across consecutive collectives. The 10 KiB all-reduce suite measures
**34.47 us mean / 34.17 us p50** (500 iterations, host synchronization included), versus the
previous implicit CUDA-staged path's approximately 48.4 us. A graph-captured Q4_K_M run with 85,000
occupied tokens, 180,000 capacity, INT8 KV, and MTP3 produced **53.18 committed tok/s** (89 accepted
/ 114 drafted, 78.07% acceptance); the matched earlier result was 53.41 tok/s, so this is a
transport-latency cleanup rather than an 80 tok/s breakthrough.

Removing the remaining PCIe dependency requires a pipeline-parallel 32/32 layer split (one hidden
transfer at the layer boundary instead of two collectives per layer). That is a runtime/state/KV/
CUDA-Graph architecture change, not a safe kernel tweak; it remains outside the current TP2
contract.

The active port uses two Tesla V100-SXM2 16 GB cards, CUDA 12.8 (`sm_70`), and the local
`qwen3.8-27b/gguf-q4-k-m` artifact converted from LM Studio's Qwen3.8-27B Q4_K_M GGUF.
On the fixed code-generation workload below, three valid runs at exactly 85,000 occupied prompt
tokens measured **53.4075 ± 0.0639 committed decode tok/s** (mean ± sample standard deviation),
**18.68% above** the user's approximately 45 tok/s baseline. Each run measures 512 decode tokens
with 180,000-token capacity. This establishes the speed result for this prompt and output window;
it does not guarantee the same speed on every prompt. The RTX 5090 campaign summary later in this
document describes inherited results for different hardware and weight profiles.

Startup chooses the transfer route by measurement, not by inference: when both devices advertise
peer access it enables it and verifies the route with exact byte comparisons in both directions
across the range of payload sizes the collectives move, and it rejects startup if verification fails.
A Linux IOMMU `DMA` or `DMA-FQ` domain appears in the diagnostic but is not by itself disqualifying.
Where peer access is unavailable or fails that verification, TP2 uses two long-lived pinned host
slots and explicit asynchronous D2H/H2D legs for every collective instead of relying on the driver's
opaque UVA-D2D staging path.
The real 10 KiB decode-shaped all-reduce measures **33.67 us mean / 33.05 us p50** over 500
iterations (down from the earlier 48.4 us host-staged measurement); this reduces transport overhead
but cannot remove the 128 per-token collective dependencies. The explicit route passes the collective
suite, including different tensor sizes, guards and 64 consecutive rounds, and the three public
Engine CUDA Graph measurements below. The INT8 attention test also passes its independent FP64
oracle at 85K occupied keys for all four queries, all 24 heads and every visible key, with TP1/TP2 comparisons
and exact cache checks. Real-model MTP/non-MTP teacher forcing and graph/eager regression gates
pass on their short-prompt fixtures. The source Q4_K/Q6_K codes and scales remain unchanged;
these numerical and state checks support the tested route without asserting universal quality
parity for all prompts.

| Setting | V100X2 comparison profile |
|---|---|
| GPUs | 2 x Tesla V100-SXM2 16 GB, TP2, devices 0 and 1 |
| CUDA compile/runtime | 12.8 / 12.8 |
| Artifact | `qwen3.8-27b/gguf-q4-k-m`; original Q4_K/Q6_K blocks and scales |
| Request mode | One active request |
| Context capacity | 180,000 tokens, native RoPE |
| Primary occupied context | Exactly 85,000 actual prompt tokens |
| KV cache | INT8 group-64; compared with LM Studio's Q8 KV configuration |
| CUDA Graph | Enabled |
| Prefill chunk | 1,024 tokens |
| NInfer MTP | Fixed draft window of three; optimized proposal head (`--lm-head-draft`) |
| Measured output window | 512 decode tokens plus the first token from prefill; three repetitions |
| TP2 transport | Verified CUDA host-staged copies on the current IOMMU host |

NInfer's `--mtp-draft-tokens 3` uses a fixed three-token proposal window, shortened when the remaining
output or context budget requires it. Verification may accept zero drafts. This is distinct from
LM Studio's maximum-three/minimum-zero draft configuration: a minimum draft count of zero permits
its proposal policy to vary how many drafts it attempts, whereas zero accepted drafts describes
the verification result. The two engines therefore use related, but different, MTP schedules.

| Repetition | Committed decode wall tok/s |
|---|---:|
| 1 | 53.3498 |
| 2 | 53.3966 |
| 3 | 53.4761 |
| Mean ± sample standard deviation | **53.4075 ± 0.0639** |

Every repetition produced the same 513 token IDs, with no EOS/EOG token in the captured window,
and accepted **364 / 444 drafts (81.98%)**. Before the current prefill route, this decode campaign's
prefill took **298.132–298.154 s** per repetition and was excluded from decode throughput.
Host CPU use was approximately one of 32 logical cores; observed GPU memory use was
**13,768 / 13,454 MiB**, including about 313 MiB of desktop use on GPU 0.

A separate short-input code-chat measurement used 512 prompt tokens, the same 180,000-token
capacity and execution settings, one warmup, and three 256-token decode windows. Its wall decode
rates were **59.9644, 60.0509 and 60.0721 tok/s**, or **60.0291 ± 0.0571 tok/s** (mean ± sample
standard deviation), with **65.89%** draft acceptance. All three 257-token outputs were identical
and EOS/EOG-free. This exceeds the reported 57 tok/s peak numerically, but the unspecified
occupancy of that peak prevents a matched comparison; it is not the 85K acceptance result.

The user-reported LM Studio baseline is approximately **45 committed decode tok/s at 85K occupied
context**, with a reported **57 tok/s peak** whose context occupancy was not specified. The
approximately 40 tok/s figure at longer context is an estimate. These observations are the
acceptance reference; the 18.68% gain is relative to the reported 45 tok/s value, rather than the
matched-engine measurement below. The 57 tok/s peak has not been exceeded by this 85K result and lacks
the context occupancy needed for a like-for-like comparison.

A matched diagnostic run before the current NInfer prefill optimization used LM Studio's CUDA
backend **2.33.0** with its automatic two-GPU split, the same source GGUF and exact 85,000
prompt IDs, Q8 KV, a requested 180,000-token capacity
(rounded by the backend to 180,224), and maximum-three/minimum-zero MTP. Both engines used greedy
sampling and generated 513 output tokens: one from prefill and 512 in the measured decode interval.

| Engine | Decode tok/s | Prefill seconds | Accepted / drafted |
|---|---:|---:|---:|
| NInfer V100X2, mean of three runs | **53.4075** | 298.144 | 364 / 444 per run (81.98%) |
| LM Studio CUDA 2.33.0, one run | **35.4977** | 185.374 | 365 / 440 (82.95%) |

The LM run decoded for **14.42347 s**, evaluated all 85,000 prompt tokens without cache reuse,
and stopped at the output limit without any EOS/EOG token. NInfer's decode rate is **50.45% higher**
in this comparison, with similar MTP acceptance. Its prefill is **1.61 times as long**, so this is
a decode improvement for that earlier implementation, not a current cold-request latency result.
The single LM run is a diagnostic, not a stable average, and does not replace the user's approximately 45 tok/s
acceptance baseline.

Measure committed output tokens per decode second, excluding the first token produced by prefill.
Rejected draft tokens do not count as output. Report repeated measurements and context occupancy;
a short-prompt result at `--max-context 180000` cannot establish performance at 85K occupied tokens.
Use the same prompt content and sampling for the final LM Studio comparison. Record GPU memory,
power and aggregate CPU use; keep CPU below the user's approximately 85% ceiling.

The public Engine benchmark provides a reproducible greedy diagnostic. The code-chat corpus below
uses the artifact's embedded tokenizer and chat template with thinking disabled, distinct repository
source excerpts, and a final bounded blocking task-queue implementation request. The tool trims the
source excerpt body to make the complete prompt exactly 85,000 tokens, preserving the task and
assistant prefix; it does not repeat excerpts to fill the context. Feed the saved token IDs directly
to both engines, without applying another template or decoding and retokenizing them. Keep the
corpus command's `--output-tokens 1024`: this is part of the fixed prompt's wording, independent of
the measured 512-token decode window. The bundled 65,536-token benchmark corpus is too short for
this case:

```bash
cmake -S . -B build-v100 -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-v100 --target ninfer_bench ninfer_v100_corpus -j

LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_v100_corpus \
  /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  /tmp/v100-code-85000.ids --code-chat 85000 --output-tokens 1024 \
  src/core/host_worker_pool.h \
  src/core/host_worker_pool.cpp \
  src/runtime/engine/concurrent_executor.h \
  src/targets/qwen3_6/impl/runtime/program_impl.h \
  src/targets/qwen3_6/impl/runtime/text_context_impl.h \
  src/targets/qwen3_6/impl/runtime/layouts_impl.h \
  src/targets/qwen3_6/impl/runtime/mtp_impl.h \
  src/ops/kernel/gqa_attention_decode_i8.cuh \
  src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh

LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_bench \
  --weights /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  --tp 2 --devices 0,1 --max-ctx 180000 --kv-dtype int8 \
  --prefill-chunk 1024 --mtp-draft-tokens 3 --lm-head-draft \
  --corpus /tmp/v100-code-85000.ids \
  -pg 85000,512 --warmup 0 -r 3 --capture-generation -o json \
  --output-file /tmp/ninfer-v100x2-85000.json
```

`-pg 85000,512` fixes the output window at 513 tokens: one from prefill and 512 from decode.
Each measured decode follows the full 85K-token prefill, and the Engine primes its CUDA Graphs
before use; `--warmup 0` omits an additional benchmark warmup generation. For a cross-engine
wall-time decode comparison, compute each repetition's rate as
`512 / (timings.total_seconds - timings.first_token_seconds)`. Both timestamps include the same
prompt preparation and submission origin, so subtraction leaves the interval from first-token
commit to request completion, including work between decode rounds. The separately reported
`decode_output_tok_s` uses accumulated Program decode-phase time and excludes some Engine work
between rounds.

`--capture-generation` retains each measured repetition's raw text and token IDs under
`reps[].generation`. NInfer's fixed-budget benchmark disables model-default stopping but can emit
EOS tokens. An accepted repetition must contain no EOS within its entire 513-token output window;
also inspect the captured text for repetition before counting its rate as useful answer throughput.
The matched 512-token LM comparison uses `compare_llama.py` with `ignore_eos=false`, without
EOG-token logit biases. It requires `tokens_predicted=513` and a non-EOS ending; otherwise it saves
the returned output and rejects the repetition instead of calculating a complete-window rate.

The comparison used this local LM Studio backend command, leaving GPU splitting automatic:

```bash
LD_LIBRARY_PATH=/home/z/.lmstudio/extensions/backends/vendor/linux-llama-cuda-vendor-v1 \
/home/z/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.33.0/llama-server \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --ctx-size 180000 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 \
  --threads 16 --threads-batch 24 \
  --host 127.0.0.1 --port 18081 --no-webui
```

With that backend ready, run the matched single-round diagnostic using Python 3.11:

```bash
.venv/bin/python3 tools/v100/compare_llama.py \
  --url http://127.0.0.1:18081 \
  --corpus /tmp/v100-code-85000.ids \
  --prompt-tokens 85000 --decode-tokens 512 --repetitions 1 \
  --output /tmp/llama-v100x2-85000.json
```

This one-round LM Studio diagnostic does not establish a stable average. The NInfer comparison
against the user's 45 tok/s baseline uses the three measured repetitions above.

These are fixed-window code-generation throughput measurements, not evaluations of whether the
generated code correctly solves the requested programming task.

Quality checks are separate from timing. Preserving the source quantization blocks is an exact
conversion claim; control-tensor transformations and floating-point operators require numerical
oracles. Real-model MTP/non-MTP teacher forcing, graph/eager comparisons and cross-device state
checks qualify the execution path. A plausible answer or a faster kernel alone is insufficient to
claim unchanged model quality or an end-to-end speedup.

## V100X2 maximum-context capacity sweep

The capacity comparison in the README varies only the requested maximum context from 1,024
through 65,536 tokens, doubling at each step. Every request uses the same 512-token code-chat
prompt and generates 257 tokens: one from prefill and 256 in the measured decode interval.
It measures the effect of the capacity setting, not inference with that many occupied tokens.

Each engine starts with a fresh resident model at each capacity, discards one complete warmup
request, and measures three complete requests with prompt-cache reuse disabled. Both consume
the exact saved prompt IDs, use greedy sampling, and retain their configured MTP3 policies.
NInfer uses TP2, INT8 group-64 KV, CUDA Graphs and the optimized proposal head; LM Studio uses
backend 2.33.0, automatic GPU splitting, Q8 KV and maximum-three/minimum-zero MTP as above.
Rates exclude the first token and loading time. The tool rejects EOS/EOG, incomplete windows,
truncated input and unexpected LM prompt-cache reuse. Reported deviations are sample standard
deviations across three repetitions.

Reproduce the prompt and comparison using the existing artifact and Python 3.11 environment:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_v100_corpus \
  /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  /tmp/v100-code-512.ids --code-chat 512 --output-tokens 1024 \
  src/core/host_worker_pool.h

.venv/bin/python3 tools/v100/bench_capacity.py \
  --corpus /tmp/v100-code-512.ids --output-dir /tmp/v100-capacity
```

The runner executes the engines sequentially and stops only its own temporary LM server.
Its output directory contains raw JSON responses, engine logs, requested and actual capacities,
per-repetition timings, MTP counts, and `summary.json` / `summary.md`. The `--engine ninfer`
and `--engine llama` options allow running the two sides separately in that same directory.

## V100X2 prefix cache

The same two V100-SXM2 16 GB cards and GGUF-derived Qwen3.8-27B Q4_K_M artifact support
retained-prefix reuse with TP2 and optimized MTP3. The real-model gate uses INT8 group-64 KV,
4,096-token capacity, 256-token prefill chunks, greedy sampling and 32 output tokens. A rendered
code prompt contains 3,274 tokens; `preserve_thinking=true` saves its complete response boundary.

Three warmed cold/cache pairs measured median Engine time to first token of **11.9119 s cold**
and **0.0173602 s cached** (686×). Model loading, prompt rendering and HTTP transport are excluded.
Diagnostic logit/peer copies are disabled for these pairs. Cached requests report 3,274 reused
tokens and zero computed prefill tokens. Two exact-history follow-up turns each prefill only 30
tokens, with roughly 0.20 s time to first token. These are prompt-work savings, not a decode-rate
increase or a cache-enabled comparison against LM Studio.

The gate covers response-checkpoint replay, repeated append, zero-suffix sampling, changed-prefix
reset, rewritten response suffixes and stopping inside an accepted MTP round before continuing.
Repeated checkpoint execution, direct continuation versus checkpoint restore with identical
prefill partitions, and CUDA Graph versus eager execution must agree exactly in generated tokens,
captured logits and MTP acceptance. Both ranks' speculative egress must agree.

Cold re-prefill uses a different BF16 GEMM/GDN partition from retained decode and suffix state.
Two cold comparisons first diverged after 26 and 25 identical output tokens respectively; at each
shared history, a fresh single-output target evaluation assigned the cached choice exactly the
same logit as its selected winner (deficit 0). The test checks this first divergence against the
existing TP2 0.5-logit near-tie bound; it does not claim bit-identical free-running output across
different prefill partitions. Other cold output comparisons remain exact.

Reproduce the Engine gate and the actual HTTP turn/response-checkpoint smoke with the existing
artifact:

```bash
export LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NINFER_V100X2_ARTIFACT=/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  build-v100/tests/ninfer_qwen3_8_27b_v100x2_prefix_real_test

.venv/bin/python3 tools/smoke/serve_thinking_preservation.py \
  --artifact /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
  --server-bin build-v100/apps/ninfer-serve --backend mtp \
  --tp 2 --devices 0,1 --kv-dtype int8
```

The HTTP smoke uses a temporary local server with a 1,024-token capacity and 128-token chunks.
The cache remains process-local and can resume only the current frontier or its saved complete
turn/response checkpoint; see [serving cache behavior](serving.md#execution-behavior).

## KV-cache dtype on the Turing target

On this target the `fp16`/`int8` choice is not a close call, and the reason is worth stating
plainly: the two values do not differ in round time at all, and the entire throughput difference is
MTP acceptance. (`fp8` is a third value; it arrived later and is measured in its own subsection
below.)

Method: one artifact, one prompt, one capacity, one draft window; only the KV dtype changes between
the paired rows. Both rows of a pair are single runs on the same idle machine, in the same session.

- Hardware and toolchain: 2 x RTX 2080 Ti 22 GB (`sm_75`), CUDA 12.8, TP2 with host-staged
  collectives, GDN state in FP32.
- Artifact: `qwen3.8-27b/groupwise-int` (registered `Q4`/`Q5`/`W8` weights), native RoPE.
- Runtime: `--tp 2 --devices 0,1 --max-context 262144`, explicit `--kv-capacity 262144`, CUDA Graphs
  on, `--spec mtp --draft-tokens 3 --lm-head-draft`, greedy, `--max-new 64`.
- Prompts: a 16,042-token and an 85,070-token prompt, each filled from cold with no prefix reuse.

| Occupied context | `--kv-dtype` | Decode | MTP acceptance | tok/round | Round time | Prefill |
|---:|---|---:|---:|---:|---:|---:|
| 16,042 | `int8` | 48.50 tok/s | 42.68% | 2.25 | 46.4 ms | 386 tok/s |
| 16,042 | `fp16` | **64.30 tok/s** | **67.74%** | **3.00** | 46.7 ms | 371 tok/s |
| 85,070 | `int8` | 39.91 tok/s | 50.00% | 2.48 | 62.1 ms | 300 tok/s |
| 85,070 | `fp16` | **45.69 tok/s** | **62.12%** | **2.86** | 62.6 ms | 254 tok/s |

Both 16-bit rows above were measured while the cache still stored BF16 words and every attention
kernel widened them to FP16 on the fly. The cache now stores the FP16 tensor-core operand directly,
with the widening moved to the write (see [choosing the KV-cache dtype](cli.md#choosing-the-kv-cache-dtype)).
That is not a precision change: the mma operands are bit-identical to the ones the kernels used to
build from the same bits, so acceptance is unchanged. Re-measured on the host-staged pair
(`--devices 0,2`, same artifact, same prompt, same flags, `--max-new 64`), the 85,070-token
acceptance workload went from 48.70 tok/s to 50.09 / 49.97 / 49.95 tok/s (+2.7%) with the MTP
acceptance rate, acceptance length, and per-position accept counts (20,14,7) all unchanged at
62.12% / 2.86. One nsys profile of that run attributes it: the small-T decode attention kernel
measures 784.1 -> 704.3 us per launch (-10.2%) over the same 950 launches, the prefill attention
kernel -12.8%, and total kernel time across the whole run -4.4%.

**That `50.09` figure is a dated step, not the current result.** On the same pair, the same
artifact, the same prompt and the same flags, this target now measures **54.31 / 54.36 / 54.32 /
54.26 committed decode tok/s** at instrumentally identical acceptance counters -- 22 MTP rounds,
62.12% acceptance, 2.86 tokens/round, per-position accept counts 20/14/7. Those four runs were
interleaved in one session with a freshly rebuilt control binary that read **53.80 tok/s** at the
same counters, which is the pair that isolates the last change (`+0.94%`, section 39 of
`history.md`); the counters are those of the `45.69` row in the table above, so the whole
`fp16`-cache era reads **45.69 -> 54.3, +19%**, and none of it moved the committed token stream. The
later changes are recorded in `history.md`; the profile behind the current bill is section 37 of
that file, re-taken after section 39.

**The `int8` rows are dated.** Re-measured at the same 85,070 occupied tokens, the same flags and
`--max-new 64`, `int8` now reads **42.05 tok/s / 43.75% acceptance / 2.30 tok/round** while `fp16`
reads 54.31 / 54.36 tok/s at 62.12% / 2.86. The dtype conclusion is unchanged -- `int8` is 22.4%
slower and 18.4 acceptance points down -- and the deficit has widened rather than closed.

Round time is derived as `tok/round / decode`; it is reported because it is the quantity that did
*not* change. At both context lengths the two dtypes agree on it within one percent, in spite of
`fp16` reading twice the KV bytes. The attention kernel is latency-bound at this width and occupancy,
not bandwidth-bound, so the halved footprint buys no time, while the `int8` staging path pays a
dequantization the `fp16` path does not. What `int8` does cost is accuracy: 15.4 acceptance points at
85,070 tokens and 25.1 at 16,042, which is a 19.5% and 37% throughput penalty respectively, and it is
larger than the round-time difference it was supposed to buy.

Two further measured facts bear on interpreting the acceptance column:

- **MTP acceptance is content-dominated.** The same `int8` KV at 85,070 tokens accepted 46.75% at
  `--max-new 128` and 50.00% at 64; a nested 32,066-token prompt accepted more (51.39%) than its own
  16,042-token prefix (42.68%). MTP numbers are comparable only at equal `--max-new` on the same
  prompt, which is why every row above is `--max-new 64` on a fixed prompt.
- **The `int8` deficit is cache precision, not a defect in the `int8` route.** Two cheaper
  explanations were tested and refuted. Routing the `int8` narrow widths (T=1/T=2) onto the same
tensor-core kernel the verify width uses left the 16,042-token acceptance bit-identical
  (42.68%, per-position accept counts 18/10/7 unchanged), so the draft and verify steps already
  agreed. A finer quantization group is not the answer either: at 64 elements per group the
  symmetric `absmax/127` scale already holds about 43 dB of SNR on a Gaussian input, and halving the
group buys roughly 1 dB, against a deficit of 15-25 acceptance points.

Consequently this checkout's default remains `fp16`, and the `int8` value is selected explicitly by
callers that need its capacity -- including the V100X2 launcher, whose own recorded measurements were
taken with `int8`.

### FP8-E4M3 KV cache on this target

`--kv-dtype fp8` stores K/V as E4M3FN codes with **no per-group scale plane**: the E4M3 exponent
covers the KV activation range directly, so quantization is one saturating cast per element and the
pool keeps the same two-plane shape as `fp16` at one byte per element. Codes are dequantized to FP16
in registers on the read path, which is why its prefill cost sits below the `int8` group-absmax/scale
path. The Op's FP64 oracle judges it against an independent host E4M3 codec; its registered
short-window criterion is `relative_l2` **2.8e-3**, against a measured **2.0-2.4e-3** -- better than
the `int8` profile's 3.15e-3, since there is no per-group scale to add a second rounding.

Method: one artifact (`qwen3.8-27b/fp8-block128`), TP2 `0,1`/NVLink, CUDA 13.2, CUDA Graphs, MTP3
with `--lm-head-draft`, greedy, one request, `.scratch/codechat_85000.ids` prefixes (real text, so
MTP acceptance discriminates). Only the KV dtype changes between rows. These rows are on a different
artifact and toolchain than the `groupwise-int`/CUDA 12.8 block above, so the two tables are not
row-comparable. Acceptance and decode are deterministic for a fixed configuration: three alternating
repeats of the 32,768-token row agreed bit-for-bit, so the single runs below are reproducible points
rather than samples.

| Occupied context (generated) | `--kv-dtype` | Prefill tok/s | Decode tok/s | MTP acceptance | KV payload |
|---:|---|---:|---:|---:|---:|
| 8,192 (512) | `fp16` | 1,271.5 | 48.82 | 0.6026 | — |
| 8,192 (512) | `fp8` | 1,211.2 | **54.63** | **0.7231** | −50% |
| 8,192 (512) | `int8` | 1,178.2 | 50.35 | 0.6456 | −50% |
| 16,384 (512) | `fp16` | 1,150.8 | 52.60 | 0.7379 | — |
| 16,384 (512) | `fp8` | 1,137.8 | **54.66** | **0.7766** | −50% |
| 32,768 (512) | `fp16` | 1,115.4 | 51.10 | 0.7297 | — |
| 32,768 (512) | `fp8` | 1,048.3 | 49.53 | 0.7010 | −50% |
| 32,768 (512) | `int8` | 982.2 | 47.46 | 0.6544 | −50% |
| 85,000 (64) | `fp16` | 878.1 | 48.19 | 0.8182 | 2.760 GiB |
| 85,000 (64) | `fp8` | 828.4 | 48.75 | **0.8182** | **1.380 GiB** |
| 85,000 (64) | `int8` | 730.6 | 48.71 | 0.8333 | 1.423 GiB |
| 140,000 (64) | `fp16` | 733.7 | 41.21 | 0.7333 | 4.543 GiB |
| 140,000 (64) | `fp8` | 669.6 | 41.49 | **0.7333** | **2.271 GiB** |

These rows are from the build that still folded the fp16 accumulator into fp32 each K=64 and still
decoded weights through the shared LUT. The current lane, at the same two long-context occupancies,
measures **1,030.2 and 808.6** prefill tok/s for FP16 KV (see "Long-context scaling on this
target" below), so the absolute columns above are superseded; the `fp8`-versus-`fp16` **deltas** are
same-build comparisons from that campaign and have not been re-measured on the current build.

**Prefill** costs 1-9% against `fp16` (rising with the window, as the dequantization work scales with
the key range), which is consistently under half of `int8`'s 7-17% at the same occupancies.
**Decode** differs only where acceptance differs; at both long-context rows it is `fp16`'s equal or
slightly ahead, because the halved KV read is not the binding term at this width.

**Acceptance is a trajectory quantity, not a pure quality number.** `--capture-generation` shows the
8,192-token row produces **token-identical output** through 129 tokens while its acceptance differs
by 12 points from `fp16`'s, so that difference is entirely draft/verify-round structure rather than
answer quality; the 32,768-token row diverges from generated index 11, where acceptance and
trajectory are no longer separable by this measurement. The two long-context rows -- the ones that
matter for this feature -- agree with `fp16` **bit-for-bit** on acceptance (0.8182 and 0.7333). A
fixed-trajectory or independent quality evaluation is still open; see `todo.md`.

**Capacity is where `fp8` changes what the machine can do.** With weights at 15.296 GiB and a
6.03 GiB runtime budget per card, an `--max-ctx` sweep puts `fp16`'s ceiling at **153,984 tokens**
(it fails at 153,990 by 0.4 MB) and `fp8` at the model's **native 262,144-token limit** with memory
still to spare -- a **1.70×** window. Both ceilings were then run end to end on real prompts:

| `--kv-dtype` | max_ctx | KV payload | prefill tok/s | decode tok/s |
|---|---:|---:|---:|---:|
| `fp16` | 153,984 | 4.993 GiB | 703.0 | 50.89 |
| `fp8` | 262,144 | **4.250 GiB** | 467.1 | 28.62 |

`fp8` holds a **1.70× longer window in 15% less KV memory** than `fp16` at its own ceiling, and
occupies 21,155 of 22,528 MiB per card at 262,144 tokens. The prefill and decode columns in that
table are different windows and must not be read against each other; at equal occupancy the
comparison is the table above. The `int8` value remains available and is still the slower, less
accurate of the two halved-footprint options at every point measured here.

### TP2 transport on this target

TP2 collectives here run over the **direct NVLink path** rather than the pinned host-staged route.
Startup selects the route by measurement, not by inference (see
[the transport rule](maintainer/tp2-yarn-1m.md#2-transport-the-route-is-measured-not-inferred)):
this pair reports an IOMMU `DMA-FQ` domain, yet it advertises peer access in both directions and
copies exact payloads at every probed size, so the direct route is chosen. The evidence is a
same-session A/B in which only the route differs:

| Configuration | Host-staged | Direct |
|---|---:|---:|
| 85,070 tokens, `fp16` | 45.53 tok/s | **46.89** (46.83 / 46.86 / 46.88 / 46.99) |
| 85,070 tokens, `int8` | 39.95 | **41.03** (41.00 / 41.07) |
| 16,042 tokens, `fp16` | 60.69 (61.18 / 60.20) | **63.38** (64.69 / 62.60 / 62.85) |
| 16,042 tokens, `int8` | 46.28 (47.17 / 45.55 / 46.13) | **46.58** (46.58 / 46.68 / 46.47) |

Prefill moves the same way, which is expected because a 4,096-token chunk's all-reduce is the
largest payload the collectives carry: at 85,070 tokens it measures 256.55 -> 268.27 tok/s for
`fp16` and 300.97 -> 317.46 for `int8`.

MTP acceptance is **bit-identical** on both routes -- 62.12% and 2.86 tokens/round at 85,070 `fp16`,
50.00% and 2.48 at 85,070 `int8`. That is structural rather than incidental: both routes deliver the
same bytes into the same staging tensor and then run the same combine, so the committed token stream
must not move. At the operator level, the same 10 KiB decode-shaped all-reduce that
`ninfer_allreduce_test` times reports 29.86 us mean / 29.59 us p50 on this build.

The machine's session spread is real but not uniform, so read the absolute values with it in mind.
At 85,070 tokens the host-staged route reproduced its earlier value closely -- 45.53 here against
45.69 in the table above, and four direct runs spread 0.17% -- while at 16,042 tokens the same
unchanged configuration moved from **64.30 tok/s** in the table to **60.69** in the A/B session. The
paired comparison inside one session is therefore the durable evidence for the transport change, and
a single absolute value at the short context is not.

The A/B above was measured **before the KV cache moved to FP16 storage**, and its scope is one fixed
device pair with only the route toggled. After that change the same 85,070-token `fp16` acceptance
workload measured **47.86 tok/s** on the direct `0,1` pair and **50.10 tok/s** on the host-staged
`0,2` pair, both at 62.12% acceptance. Since then `0,2` has moved to **54.31 / 54.36 / 54.32 / 54.26
tok/s** at the same acceptance counters; `0,1` was not re-measured. Those two rows differ in the card pair as well as in the
route: `0,1` is the NVLink pair and contains this machine's slowest card, `0,2` is host-staged and
avoids it. The route conclusion here is therefore about a fixed pair, and it does not say that the
host-staged route is the faster one overall -- on these three cards it is not.

The pair matters in both directions, and not only through the slow card. On a 32,066-token prompt
the same build prefills at **1,220.70 tok/s** on `0,1` against **1,005.6** on `0,2` -- 21% higher on
the NVLink pair -- while decoding within 2% of it (73.99 against 72.64 tok/s). A 4,096-token
prefill chunk's all-reduce is the largest payload the collectives carry, so the NVLink route wins
wherever prefill dominates; `0,2` is the better pair for long-context decode, where the slow card's
per-launch cost is what sets the round.

### SM75 block-FP8 head optimization

The official block-FP8 artifact's full vocabulary head is contiguous BF16. Its former SM75
route used the general MMA launcher even at T1-T4. The BF16 Op now selects the existing GEMV
body at T1 and exact small-T body at T2-T4 for both `248320x5120` and TP2 `124160x5120`;
wider columns and other architectures keep the MMA path. No family schedule, weight bytes,
allocation or CUDA Graph ownership changed.

Measured on this host with CUDA **13.2.51**, driver 595.91.07, two RTX 2080 Ti 22 GB cards,
TP2 `0,1`/NVLink, FP16 KV, capacity 100000, chunk 4096, MTP3, optimized draft head and graphs.
The fixed corpus is `.scratch/codechat_85000.ids`; `-pg 32768,512 -r 1 --warmup 1` was run
in same-binary MMA/narrow/narrow/MMA order using a temporary switch removed after measurement:

| Route | Prefill tok/s | Committed decode tok/s | Rounds / Acceptance / Fallbacks |
|---|---|---|---|
| former MMA | 1025.99 / 1021.60 | 44.11 / 43.94 | 161 / 0.7297297297 / 0 each |
| narrow head | 1022.50 / 1031.05 | 50.97 / 50.94 | 161 / 0.7297297297 / 0 each |

The mean committed decode gain is **15.7%**; there is no prefill-gain claim. A GPU1 cold-cache
`124160x5120` Op comparison at T1-T4 measured approximately 9.65-9.76 ms versus 2.18-2.20 ms.
Every route is checked against the independent FP64 BF16 Linear oracle (T1 all rows,
T2-T5 sampled outputs) without changing the reduction criterion. The real FP8 TP2/FP16 MTP
check also passes repeated generation, graph/eager equality, peer egress and teacher forcing;
the one near-tie disagreement is also present in the ordinary-decode control.

These are current CUDA 13.2 measurements, not requalification of the historical CUDA 12.8
rows or an external vLLM A/B. The 85K workload and full source-aligned model quality were not
remeasured. Details are in [history section57](../history.md#57-2026-10-05-sm75-bf16-全词表-head-窄-t-路由).

### External SM75 reference: `vLLM-2080Ti-Definitive`

The external same-GPU-class reference is
[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive), an SM75-focused
vLLM fork for dual RTX 2080 Ti (and for Tesla T10/T40/T4, TITAN RTX and Quadro RTX 6000/8000).
Its [2x2080Ti profile](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/main/profiles/2x2080Ti/README.md)
Notes 5 records a 2026-09-19 v0.2.1 test on dual Xeon E5-2673 v4, physical GPUs `1,5` with NV2,
CUDA 13.0, Torch 2.13 and driver 595.91.07. This checkout's recorded host is EPYC 7352 with
CUDA 12.8. No local reproduction of those external rows has been established; a separate local run
of the same stack, which measures concurrency rather than those rows, is recorded in
[the external reference](maintainer/vllm-2080ti-definitive-reference.md#83-local-concurrency-reproduction-on-this-host-2026-10-07).
Similar prompt length and GPU model do not establish identical prompt content, hardware or
execution conditions.

| Route | Prefill | Decode |
|---|---:|---:|
| external FP8 weights, FP8 KV, MTP4, 32K/512 | 1,393.94 tok/s | 92.47 tok/s |
| external FP8 weights, FP16 KV, MTP4, 32K/512 | 1,425.34 | 95.10 |
| external FP8 weights, FP16 KV, no MTP, 32K/512 | 1,483.04 | 30.80 |
| local groupwise-int on `0,1`, 32066/512 | 1,220.70 | 73.99 |
| local groupwise-int on `0,2`, same local flags | 1,005.60 (1,006.77 / 1,004.50) | 72.64 (72.63 / 72.65) |

Both groupwise-int checkout rows are one command -- 32,066 prompt tokens, 512 generated, `--tp 2 --max-context
262144 --kv-dtype fp16 --spec mtp --draft-tokens 3 --lm-head-draft --greedy` -- and it reports 156
MTP rounds and 75.64% acceptance at every point. The earlier `1,124.96 / 59.81` row did not record
its pair or KV dtype; those cannot be reconstructed from similarity in throughput. These are
single runs or pairs of runs, not campaigns.

This is not a matched performance or quality comparison. In particular, the external FP16-KV row
is faster than its FP8-KV row, so attributing the local gap mainly to FP8 KV is unsupported.
The 209-220+ tok/s DFlash2 headlines use NVFP4 and synthetic high-acceptance inputs, not this FP8
MTP workload. Marlin packs compressed FP8 words and permutes/compensates scales; its inner loop
still performs `dequant_data -> scale -> mma` on register fragments. It does not retain a
full-model FP16 expansion. A local same-source Linear comparison now measures approximately
1.5x Marlin operator speed on two real TP2 MLP prefill shapes, with both routes checked against
an independent FP64 oracle. It uses installed vLLM 0.26, not the external profile version,
and does not yet establish an Engine improvement; see
[the operator evidence](../history.md#58-2026-10-05-同机同源-marlin-linear-对照).

Two things were taken from that stack, and both are recorded here as evaluated rather than assumed:

- **The prefill accumulation form.** Under `__CUDA_ARCH__ == 750`, the Marlin W8A16 route sets
  `use_fp16_accum = true` (`marlin_template.h`: fp16 activation, fp8 weight, not NVFP4, not the
  ungrouped 4-bit case), so its `mma.sync.m16n8k8.f16.f16.f16.f16` runs a **pure fp16 accumulator
  for the whole K with no fp32 reduction across segments**. (Those files are upstream vLLM/Marlin
  code -- the fork's own contribution there is lowering the Turing gates on the Python side so the
  routes run on `sm_75` at all.) This checkout's block-FP8 prefill HMMA now matches that form:
  removing the per-segment fp32 fold measured **+10.3% prefill** in same-session alternating A/B
  (swiglu leaf 44.9 to 54.7 TFLOPS, +21.9%), with the argmax gate bit-identical to the folded
  engine; see [history section98](../history.md#98-prefill-gemm去掉-fold纯-fp16-累加对齐对手-marlin-use_fp16_accum10-prefill2026-10-07).
  It changes the accumulation order only; the weight and activation operands are bit-identical to the
  FP32-accumulation route. Its vendored CUTLASS `m8n8k16.s8` dispatch independently corroborates the
  int8 tensor-core decomposition this checkout uses for the `int8` KV path and for QK^T.
- **A GDN alternative, evaluated and not pursued.** The companion
  [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75) forward-only GDN kernel is
  a concrete alternative to this checkout's chunked WY/UT formulation, and it reports a per-stage
  gain of 2.08-2.1x. At GDN's measured share of a whole request that works out to roughly **+7.17%
  prefill and +0.61% decode** as an estimate for that historical profile, not a measured local
  replacement. It remains deferred, not disproved for every future workload.

**The external kernel's own rate on this target was later measured directly rather than inferred.**
Rebuilding its Marlin layout inside its own environment (`pack_fp8_to_int32` -> `marlin_pad_qweight`
-> `gptq_marlin_repack`; scales through `marlin_permute_scales` and
`fp8_fused_exponent_bias_into_scales`) and calling `ops.marlin_gemm` at this checkout's per-rank
prefill shapes -- with `ncu` confirming it launches the same `Marlin<...,256,4,16,4,0,2,8,0>`
variant the external 32K trace shows -- gives, at M=2048:

| `K x N` per rank | TFLOPS | `K x N` per rank | TFLOPS |
|---|---:|---|---:|
| `5120x17408` (gate/up shard) | 50.21 | `5120x8192` (GDN in) | 58.73 |
| `8704x5120` (down shard) | 58.13 | `5120x3584` (QKV) | 57.84 |
| `5120x34816` (unsharded gate/up) | 55.04 | `3072x5120` (attn out) | 55.71 |

At M=4096 the same set spans 55.00-59.45. In a same-session alternating A/B on the `17408x5120`
shard the external kernel takes 7.256 ms where this checkout's fused SwiGLU leaf takes 7.94 ms for
the same shard: **1.09x**, against the 1.22-1.5x the vLLM-0.26 operator rows above report for a
different stack. The apparent movement is the **clock**, not the structure. These cards idle at
1350 MHz and boost to 1740-1845 MHz under load, so a per-call time recorded in one boost state is
not comparable to one recorded in another: 7.256 ms at the idle clock is the same measurement as
5.60 ms at 1.75 GHz. An earlier local note attributed an equivalent sample to a "~65 TFLOPS"
external kernel; that number is this ratio re-expressed at a boosted clock, not a property of the
kernel, and the external 32K trace contains no Marlin call above 3,617 us.

**The tensor-core accumulate width is the one genuinely structural 2x on this card.** A
microbenchmark of the two Turing-native `m16n8k8` forms, eight independent accumulator chains per
warp over 68 CTAs, measures **91.0 TFLOPS** for `f16.f16.f16.f16` against **45.6 TFLOPS** for
`f32.f16.f16.f32` -- a **2.00x** plateau. `ncu`'s `sm__inst_executed_pipe_tensor` counts those PTX
ops 1:1, so 0.488 of the architectural 0.5 places the probe 97.6% up the pipe. This is the other
half of why the fp32-fold removal above was worth +10.3%: it moved this target onto the taller of
the two plateaus. Both this and the external-rate measurement are recorded in
[history.md](../history.md) as sections 104-106.

The wider map of that project -- build and launch pipeline, Turing patch surface, MTP/DFlash2
and TurboQuant algorithms, and the host behind every published number -- is recorded in
[the external stack reference](maintainer/vllm-2080ti-definitive-reference.md).

### Local block-FP8 research

Identity `qwen3.8-27b/fp8-block128` preserves the official E4M3/BF16 block128 weights and uses
SM75 W8A16 kernels. The current artifact is 30,610,987,520 bytes. The following existing results
are from [history.md](../history.md), section 54: TP2 `0,1`, NVLink, FP16 KV, capacity 100000,
chunk 4096, CUDA Graphs, MTP3 with `--lm-head-draft`, one request, `-r 2 --warmup 1`, raw IDs from
`.scratch/codechat_85000.ids`.

| Workload | Prefill tok/s | Committed decode tok/s | Recorded rounds / fallbacks | Acceptance |
|---|---:|---:|---|---:|
| `-pg 32768,512` | 1065.24 ± 30.51 | 44.60 ± 0.17 | 322 / 0 | 0.7297 |
| `-pg 84992,64` | 847.27 ± 1.04 | 44.90 ± 0.08 | 36 / 0 | 0.8679 |

Earlier prose85K/64 runs produced 31.9512/31.8412 committed tok/s, 80.12/80.40 ms/round,
25 rounds and 39/74 accepted drafts per run (52.7027%, 2.56 tok/round). They are a different
corpus, not an A/B against the codechat rows. The groupwise-int 262144-capacity/54 tok/s
acceptance is also a separate identity. FP8 quality and performance acceptance remain open.

The established local optimization evidence includes LUT plus barrier repair (same-session 32K
prefill 767 to 1072 tok/s, +39.7%) and scalar activation-load widening (prose85K approximately
114 to 80 ms/round at unchanged MTP counters). The old 67.6% scalar-kernel share describes the
114 ms round; post-fix decode attribution has not been closed. Resident bytes divided by round
time does not measure DRAM utilization, and a fixed-kernel decode/staging ablation cannot bound
all new layouts. GEMM marginal-cost experiments cannot be transferred to attention.

Use same-corpus, same-configuration alternating A/B: same-binary prefill drift reached 35% in
historical sessions. Record occupancy separately from capacity and committed throughput separately
from draft throughput. Numerical gates must cover forced scalar/HMMA routes and supported state
transitions; a successful diagnostic probe or a plausible answer is insufficient. The capture
ordering concern and concrete next checks are tracked in [todo.md](../todo.md).

### Long-context scaling on this target

The 85K and 140K occupancies are where a 22 GB Turing pair actually spends its KV budget, and
[history.md](../history.md) sections 105-106 measure them on both stacks **with the same prompt
content**: each side received `" the"` repeated to the target token count (the external tool's own
`--pure-filler`; the local side got equivalent generated prompt files), so no content difference
enters the comparison. Both stacks ran FP16 KV, TP2 `0,1`, `max_ctx` 147,456, CUDA Graphs, one
request, greedy, 128 generated tokens and no prefix reuse (`reused`/`cached prompt tokens` = 0 on
both). The external rows use its shipped
`2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env`; the local rows use the current
block-FP8 lane. MTP depth differs (external MTP4, local MTP3 with `--lm-head-draft`), so the two
decode columns are compared as **round time**, never as tok/s.

| Occupied tokens | Stack | Prefill tok/s | Decode tok/s | Tokens/round | Round ms | Base round ms |
|---:|---|---:|---:|---:|---:|---:|
| 85,070 | external, MTP4 | 1,128.6 | 99.4 | 4.704 | 47.34 | — |
| 85,122 | local, MTP3 | 1,001.9 | 42.4 | 3.040 | 71.77 | 57.40 |
| 138,933 | external, MTP4 | 888.4 | 100.0 | 4.704 | 47.05 | — |
| 138,985 | local, MTP3 | 809.8 | 37.2 | 2.963 | 79.69 | 64.06 |

- **Prefill degrades less locally, not more.** From 85K to 139K the local lane falls 19.2%
  (1,001.9 to 809.8 tok/s) against the external 21.3% (1,128.6 to 888.4). The local/external ratio
  therefore *narrows* with length, 0.888 to 0.911: the length-growing term is attention, while the
  weight-bound term the external Marlin leads on is a fixed share of each chunk.
- **Decode degrades more locally.** External round time is flat across the same span (47.34 to
  47.05 ms) while the local round grows 11.0% (71.77 to 79.69 ms), and essentially all of that is
  the base round: 57.40 to 64.06 ms without speculation, against a draft marginal of 14.4 to
  15.6 ms. External tokens/round is length-independent (4.704 at both occupancies); local MTP
  acceptance falls with length (0.6800 to 0.6543), so both effects push the same way.
- **Magnitude, and where it is not.** The local marginal length cost is approximately
  **0.125 ns per context token**. The FP16 KV byte floor for the same span is
  34.8 KB/token/card / 616 GB/s = approximately **0.055 ns/token**, so roughly 3.0 ms of the 6.7 ms
  round growth is unavoidable KV bytes and the remaining ~3.7 ms is length-dependent work that is
  not bandwidth. The external round time does not move under the same +3.0 ms byte increment.
- Local same-corpus prose rows at the same occupancies, for reference: 1,030.2 and 808.6 prefill
  tok/s, 47.2 and 43.4 decode tok/s, MTP acceptance 0.7876 and 0.8257. The two corpus sets are
  different workloads and are not an A/B pair.

Two methodology traps were found while taking these rows. First, the external stack's FP16-KV
ceiling on this host is **147,696 tokens** -- a 148,480 request fails at load by 0.01 GiB -- slightly
below this checkout's 153,984, so the two are not interchangeable above that point. Second, with
prefix caching left enabled the second long request reuses the first one's filler prefix and reports
an artefactually fast prefill (time-to-first-token moved only 77.2 to 79.2 s from 85K to 139K); its
runs here disable prefix caching explicitly. The external profiling tool also defaults to a
time-to-first-token read timeout near 90 s, which 139K exceeds.

## Concurrent decode on the Turing target

A decode round is formed over every active request, so a round's width is
`concurrency × (1 + draft)` and not the single-request `1 + draft`. Three TP2-sharded operator
families -- attention input projection, GDN input projection, and the q5 residual projection -- had
their "small-T" specialization edge written as the single-request figure `6`. Every wider round
therefore fell through to the prefill tiling: a 32×64 tile or a 64×16 MMA tile pays a whole tile
per real column. `nsys` on the same two-request workload shows the flip directly -- `q5`
per-round call counts are unchanged (≈106 vs ≈107) while the selected kernel changes from
`q5_rowsplit_gemm_simt_kernel` (91 µs/launch) to `q5_rowsplit_gemm_mma_kernel` (305 µs/launch,
**3.3×**), and `rowsplit_grouped_mma_kernel` rises from 64 to 1920 launches. Attention was never
involved: the C=2 decode attention kernel is still
`gqa_attention_small_t_tc_volta_partial_kernel<TokenTile=4, MultiBatch=1>`. The three edges now
cover the concurrency widths, and round time is linear in the batch width at
**≈10.5 ms per column**:

| Draft | Concurrency | Columns | Round | Aggregate decode | Speedup |
|---|---:|---:|---:|---:|---:|
| MTP3 | 1 | 4 | 41.8 ms | 54.6 tok/s | 1.00× |
| MTP3 | 2 | 8 | 83.1 ms | **55.4 tok/s** | 1.01× |
| MTP3 | 3 | 12 | 126.3 ms | **56.5 tok/s** | 1.03× |
| MTP3 | 4 | 16 | 160.1 ms | 61.8 tok/s | 1.13× |
| MTP3 | 6 | 24 | 218.9 ms | **65.2 tok/s** | 1.19× |
| off | 1 | 1 | 28.6 ms | 34.9 tok/s | 1.00× |
| off | 2 | 2 | 33.0 ms | 60.6 tok/s | 1.74× |
| off | 3 | 3 | 38.7 ms | **77.6 tok/s** | 2.22× |

Before the fix the same measurement gave MTP3 rounds of 125.2 ms at C=2 and 143.2 ms at C=3, i.e.
37.6 and 49.6 aggregate tok/s. The `MTP off` rows are unchanged point for point
(28.62/28.63, 32.97/32.98, 38.66/38.68 ms) because their column counts never crossed the old edge;
they are the control that shows the comparison was run in one environment. Acceptance is
bit-identical at MTP3 C=1 (drafted 2658, accepted 1160, 43.64%), and the wider concurrency points
move only within implementation-profile noise (C=3 accepts exactly the same 3548 tokens).

**Concurrency no longer costs throughput.** From four to twelve columns the aggregate stays at
54.6-56.5 tok/s, which is the single-request rate: a longer round buys proportionally more committed
tokens and costs proportionally more time. The 16- and 24-column rows read higher (61.8 and 65.2)
partly because the acceptance measured on those batch compositions is itself higher (48.9% and 46.4%
against 43.6% at C=1-3), so they must not be read as batching being intrinsically faster. Where
aggregate throughput does rise on this target is shorter draft windows, not more requests --
`MTP off` at C=3 (77.6) and MTP1 at C=2 (81.6) both beat every MTP3 width. The quantity that decides
this is **per-column token yield**: at 43.6% acceptance a verify column returns 0.44 committed
tokens, while a plain decode column returns one.

Method: `tools/bench/run_serve_concurrency.py --suite decode-saturation --mode mtp3 --mode mtp0
--concurrency 1 --concurrency 2 --concurrency 3 --decode-tokens 2048 --max-context 16384
--kv-capacity auto --kv-dtype fp16 --tp 2 --devices 0,2 --device 0`, the same 335-token prompt at
every point, one server process per point, counting only complete intervals whose decode batch
equals the configured concurrency. Kernel-level evidence and the refuted alternative explanations
are recorded in `history.md` §22-23.

The edge those three families now use (small-T specialization through 12 columns) was checked at the
same column count rather than extrapolated: forcing 16 and 24 columns back onto the SIMT
specialization costs **6.4%** (170.40 vs 160.13 ms) and **22.5%** (268.06 vs 218.91 ms) against the
MMA routes, so the partition is measured, not chosen for symmetry. Round time stays linear across
the whole range — 4/8/12/16/24 columns at 41.8 / 83.1 / 126.3 / 160.1 / 218.9 ms, i.e. 10.4 down to
9.1 ms per column as the fixed per-round cost amortizes. Note that this last figure must not be read
across column counts; only same-width A/B separates a route from an amortization effect.

## Inherited RTX 5090 campaigns (archived)

Every campaign from here to the end of this document was measured on the **upstream RTX 5090
target**: one 32 GiB RTX 5090 per single-GPU point (two for the TP2 campaign), CUDA 13.1/13.3,
INT8 group-64 KV, the `nvfp4` and `groupwise-int` registered artifacts, and that generation's
NVFP4/DFlash2/MTP3 execution paths. This checkout is a different machine and a different
implementation -- `sm_75` on 2 x 2080 Ti 22 GB, CUDA 12.8 (13.2 on the FP8 research lane), its own
attention, MTP and artifact paths, and the identities listed in the README. These rows are
inherited evidence, not measurements of this target: do not quote them as this target's
performance, and do not compare them against this target's numbers.

The detailed method, per-fixture tables and reproduction commands were removed in the documentation
compression; the machine-readable records remain under [`eval/results/`](../eval/results), and the
TP2/YaRN design itself is in
[Dual-GPU (TP2) execution and YaRN 1M context](maintainer/tp2-yarn-1m.md). What each campaign
established, in one line:

| Campaign | Headline |
|---|---|
| Single-request corpus: MTP0 long NIAH plus the MTP3 long-reasoning and cross-scenario fixtures, five seeds each | MTP3 decode 151-231 tok/s on 27B NVFP4 and 620-726 tok/s on 35B-A3B; 35B-A3B MTP0 prefill 15.5k tok/s at 7.7k tokens |
| NVFP4 concurrent corpus makespan, 75 requests at C=1/2/4/8 | C=4 finishes the corpus fastest (1,647.7 s, 2.83x C=1); C=8 is memory-bound |
| MTP3 decode saturation on `long_decode_aime26_15`, 8,192-token budget | 27B NVFP4 reaches 1,146.9 tok/s aggregate at C=8 (5.67x C=1) |
| Dual-GPU TP2 with YaRN x4 at a 1,048,576-token window, 400 W and 575 W per-GPU conditions | like-for-like MTP3 at 1M: 100.54 against 46.11 tok/s = **2.18x** |
| Cross-engine against vLLM 0.25.1, byte-identical 250k/653k/700k prompts, 500 W per GPU | vLLM prefill leads 1.17-1.32x; NInfer decode leads 2.83x/2.52x past vLLM's native window, where its drafter accepts 0% |

One result from that generation was carried into this checkout's design and is already recorded
above: the prefill accumulation form (fp16 accumulation inside a segment, fp32 reduction across
segments) under [the external SM75 reference](#external-sm75-reference-vllm-2080ti-definitive).

### EvalScope reasoning accuracy for Qwen3.6-27B (inherited)

Both weight profiles were evaluated through the same OpenAI-compatible serving route on the 5090
host with thinking enabled, MTP3, a 262,144-token context limit and EvalScope 1.9.0 (0-shot,
rule-based scoring, one sample per problem, temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, seed 42); all 258 samples completed for each profile. These are single-sample results, not
pass@k scores, and they were not re-run on this checkout.

| Weights ID | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `groupwise-int` | 86.67% (26 / 30) | 93.33% (28 / 30) | 86.87% (172 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 93.33% (28 / 30) | 84.34% (167 / 198) |
