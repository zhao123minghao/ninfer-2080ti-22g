# NInfer 2080Ti 22G

This fork targets **Qwen3.8-27B text inference on two NVIDIA RTX 2080 Ti 22 GB cards** (`sm_75`,
Turing). The qualified **groupwise-int** configuration runs tensor parallelism across both cards at the
registered native **262,144-token context capacity**, with a **FP16** KV cache, CUDA Graphs, and MTP
with up to three draft tokens and the optimized proposal head. Accepted draft counts may be zero
through three in each round.

The current build tree uses **CUDA 13.2** on `sm_75`; the Volta sibling still builds with CUDA 12.8.
The active research lane is the official **`qwen3.8-27b/fp8-block128`** checkpoint on TP2 `0,1`,
chunk 4096, with both FP16 and FP8-E4M3 KV measured. Its capacity, quality and performance evidence
are separate from the groupwise-int results below; see [FP8 research](#block-fp8-research), the
[FP8-E4M3 KV measurement](docs/performance.md#fp8-e4m3-kv-cache-on-this-target) and
[open investigations](todo.md).

NInfer is a from-scratch C++/CUDA engine descended from
[Neroued/ninfer](https://github.com/Neroued/ninfer). This checkout adapts the dual-GPU TP2 path to
Turing: it inherits the RTX 3060 fork's TP2 transport and the Volta implementation from
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100), and adds the Turing execution
leaves, the `--kv-dtype fp16|int8|fp8` cache contract, and the measurements below. Ampere (`sm_86`) and
early Ada (`sm_89`) builds remain available, and the Volta `sm_70` build is retained for the sibling
V100 checkout. The external point this checkout is measured against is the SM75 vLLM stack
[`weicj/vLLM-2080Ti-Definitive`](https://github.com/weicj/vLLM-2080Ti-Definitive), which is credited
in [Acknowledgements](#acknowledgements). See [NOTICE](NOTICE) and the
[TP2 architecture reference](docs/maintainer/tp2-yarn-1m.md) for upstream attribution and design.

## Acceptance on this target

The groupwise-int acceptance workload is **one active request on two cards**, **85,070 occupied prompt tokens**,
`--max-new 64`, TP2, MTP3 with `--lm-head-draft`, greedy sampling, CUDA Graphs and
`--max-context 262144`. Occupied tokens are stated explicitly because capacity is only an allocation
limit, never a claim about how much prompt was actually processed.

| `--devices` | Cross-device route | Decode | Prefill | MTP acceptance | tok/round |
|---|---|---:|---:|---:|---:|
| **0,2** | host-staged | **54.31 / 54.36 / 54.32 / 54.26 tok/s** | 847.09 tok/s | 62.12% | 2.86 |
| 0,1 | direct peer-to-peer | 47.86 tok/s (not re-measured) | 967.02 tok/s | 62.12% | 2.86 |

Both rows are the same artifact, prompt and flags. The `0,2` row is that pair re-measured in one
session as four runs interleaved with a freshly rebuilt control, which read 53.80 tok/s (see
`docs/performance.md`); the `0,1` row is the earlier session's value and was not re-measured on this
artifact (`0,1` carries the separate `fp8-block128` lane, whose FP16 and FP8 KV numbers are in the
KV-dtype paragraph below). The historical LM Studio reference was about **45 committed decode
tok/s** at roughly the same occupied context, but used the V100/Q4_K_M/INT8-KV workload. It is not a
matched 2080 Ti comparison or proof of equal quality. Its reported 57 tok/s peak also lacks an
occupied-context count.

The two pairs trade off different things. `--devices 0,1` is the NVLink pair and takes the direct
peer-to-peer route, which is the faster one for prefill; `--devices 0,2` is host-staged, but this
machine's card 1 runs at a lower boosted clock than cards 0 and 2, and that card sets the decode
round time. Avoiding it wins more than the slower collectives cost. **MTP acceptance is bit-identical
across the two routes** -- 62.12%, 2.86 tok/round, per-position accept counts 20/14/7 -- which is
structural: both routes deliver the same bytes into the same staging tensor before the same combine,
so the committed token stream cannot move.

`--kv-dtype fp16` is the default and the dtype used above. Measured at the same 85,070 occupied
tokens and the same flags, `int8` reads **42.05 tok/s** with **43.75%** acceptance and 2.30
tok/round: **21.9% slower** than `fp16` and **18.4 acceptance points lower**. `fp8` (E4M3 codes,
no scale plane) carries the same halved footprint without that deficit: at 85,070 and at 140,000
tokens it matches `fp16`'s acceptance **bit-for-bit**, reads back within about 1% of `fp16`'s
decode rate, and raises this target's usable window from **153,984 to the model's native
262,144 tokens (1.70x)**. The halved footprint is therefore selected through `fp8` on this target,
not `int8`. The full comparison, the "acceptance is a trajectory quantity" caveat and the two
refuted explanations for the `int8` deficit are in
[KV-cache dtype on the Turing target](docs/performance.md#kv-cache-dtype-on-the-turing-target) and
[FP8-E4M3 KV cache on this target](docs/performance.md#fp8-e4m3-kv-cache-on-this-target).

These are measurements of one machine and the registered artifact below; they are not general
targets for other hardware, artifacts, prompt content or draft windows.

## Block-FP8 research

The local artifact preserves the official checkpoint's E4M3 codes and BF16 128x128 block
multiplier scales. This is W8A16 execution on SM75, not the unsupported FP8 A8 leaf.
Recorded public-Engine measurements use TP2 `0,1` (NVLink), FP16 KV, capacity 100000, prefill
chunk 4096, CUDA Graphs, MTP3 and `--lm-head-draft`:

| Raw-token codechat workload | Prefill tok/s | Committed decode tok/s | MTP acceptance |
|---|---:|---:|---:|
| 32768 prompt / 512 decode tokens | 1065.24 ± 30.51 | 44.60 ± 0.17 | 0.7297 |
| 84992 prompt / 64 decode tokens | 847.27 ± 1.04 | 44.90 ± 0.08 | 0.8679 |

These are the existing two-repeat results from [history.md](history.md), section 54, not new
measurements. The same identity carries the FP8-E4M3 KV measurements at 8K-140K occupied tokens in
[the FP8-E4M3 KV section](docs/performance.md#fp8-e4m3-kv-cache-on-this-target). Earlier prose85K
results of about 31.9 tok/s had 0.527 acceptance; different corpora are not an optimization A/B. The
benchmark requests one additional output token in prefill;
registration and exact artifact preservation do not establish
end-to-end quality. Capture ordering is now fixed with a reproduced embedding-row exactness
regression, and representative scalar/HMMA/auto route boundaries pass independent FP64 checks.
These new diagnostics used the current CUDA 13.2 build, not the historical CUDA 12.8 performance
build. Real-activation/model-quality validation and post-fix decode attribution remain open;
[todo.md](todo.md) gives code locations and completion criteria.

The SM75 BF16 vocabulary head now uses GEMV at T1 and exact small-T kernels at T2-T4 instead
of the general MMA route. A CUDA 13.2 same-binary alternating A/B on the same 32768/512
codechat workload measured **44.11/43.94 to 50.97/50.94 committed decode tok/s (+15.7%)**,
with 161 rounds, 0.7297297297 acceptance and zero fallbacks in every run. This does not update
the historical 85K or CUDA 12.8 rows above. See [head measurements](docs/performance.md#sm75-block-fp8-head-optimization).

## Concurrent serving on two cards

A decode round is formed over every active request, so its width is
`concurrency × (1 + draft)` — not the single-request `1 + draft`. Three TP2-sharded operator
families (attention input projection, GDN input projection, and the q5 residual projection) had
their "small-T" specialization edge written as the single-request figure `6`. Every wider round
therefore fell through to the prefill tiling, where a 32×64 or 64×16 tile pays a whole tile per
real column; two active requests at MTP3 (eight columns) ran **3.3× per launch** on the q5 decode
GEMM. The three edges now cover the concurrency widths and the round time is linear in the batch
width, with the per-column cost amortizing from **10.4 ms at four columns down to 9.1 at 24**:

| Mode | C | Round | Aggregate decode | Speedup |
|---|---:|---:|---:|---:|
| MTP3 | 1 | 41.8 ms | 54.6 tok/s | 1.00× |
| MTP3 | 2 | 83.1 ms | **55.4 tok/s** | 1.01× |
| MTP3 | 3 | 126.3 ms | **56.5 tok/s** | 1.03× |
| MTP3 | 4 | 160.1 ms | 61.8 tok/s | 1.13× |
| MTP3 | 6 | 218.9 ms | **65.2 tok/s** | 1.19× |
| MTP off | 3 | 38.7 ms | **77.6 tok/s** | 2.22× |

**Concurrency no longer costs throughput.** At four to twelve columns the aggregate is 54.6-56.5
tok/s, i.e. the single-request rate: a longer round buys proportionally more committed tokens and
costs proportionally more time. The two widest rows read higher (61.8 and 65.2) partly because the
acceptance measured on those batch compositions is itself higher (48.9% and 46.4% against 43.6% at
C=1-3), so they are not evidence that batching is intrinsically faster. Where aggregate throughput
does rise on this target is shorter draft windows, not more requests -- MTP off at C=3 (77.6) and
MTP1 at C=2 (81.6) both beat every MTP3 width. The quantity that decides that is **per-column token
yield**: at 43.6% acceptance a verify column returns 0.44 committed tokens, while a plain decode
column returns one.

Measured with `tools/bench/run_serve_concurrency.py --suite decode-saturation --mode mtp3 --mode
mtp0 --concurrency 1 --concurrency 2 --concurrency 3 --decode-tokens 2048 --max-context 16384
--kv-capacity auto --kv-dtype fp16 --tp 2 --devices 0,2` — the same 335-token prompt at every
point, one server per point, only complete `decode batch == C` intervals counted. See
[concurrent decode on this target](docs/performance.md#concurrent-decode-on-the-turing-target) and
`history.md` §23 for the kernel-level evidence.

## Model artifacts

| Model | Weights | NInfer artifact | Size | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 bytes (17.07 GiB) | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 18,210,531,328 bytes (16.96 GiB) | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 bytes (20.02 GiB) | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| Qwen3.8-27B GGUF Q4_K_M (local conversion) | `gguf-q4-k-m` | `qwen3_8_27b_q4_k_m.ninfer` | local conversion | local artifact |
| Qwen3.8-27B official block-FP8 (local research) | `fp8-block128` | `qwen3_8_27b_fp8_block128.ninfer` | 30,610,987,520 bytes | local artifact |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 bytes (21.22 GiB) | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |

Qwen3.6-27B exposes two registered weight profiles, and Qwen3.8-27B adds local GGUF-derived and
block-FP8 profiles to its two published profiles. The version-2 artifact
identity selects the profile without a separate runtime flag; Qwen3.8 uses target key
`qwen3_8_27b` while sharing the 27B execution package. The Qwen3.6 `nvfp4` profile uses W4A4 Tensor
Core MMA for prefill and A16 NVFP4 kernels for decode. The Qwen3.8 `nvfp4` profile preserves its
source's mixed allocation: NVFP4 MLP weights in Text layers 0–55 and row-scaled FP8 for the token
embedding, attention input/output projections, GDN Q/K/V/Z and output projections, output head, and
remaining MLP weights. All four 27B artifacts retain the same Text, Vision, MTP, prefix-reuse, CLI,
and serving routes.

The local `gguf-q4-k-m` profile comes from the Qwen3.8 Q4_K_M GGUF in LM Studio's model directory.
Its original Q4_K/Q6_K codes and embedded scales are preserved without requantization. Control
tensors and frontend resources are converted to NInfer's representation; those transformations
require numerical and behavioral checks before making an end-to-end quality claim. See the
[GGUF artifact contract](docs/maintainer/qwen3.8-27b-artifact.md#14-preserved-gguf-q4_k_m-artifact).
It supports Text and MTP through the same Engine route; its embedded GGUF Vision objects are
validation-only and `--vision` is rejected for this identity. The artifact is intentionally kept
outside the repository because it is an 18 GB generated model file.

The independent `fp8-block128` identity preserves the official block-FP8 Text and MTP matrices;
embedding, full output head and MTP input projection are BF16. It is not the row-scaled FP8
allocation inside `nvfp4`. The local source and conversion contract are documented in the
[FP8 artifact reference](docs/maintainer/qwen3.8-27b-artifact.md#15-official-block-fp8-artifact).

## Inherited results are not this target's results

The maximum-context capacity sweep against LM Studio's CUDA backend, the DFlash2 llama.cpp
comparison, the RTX 5090 campaigns, the YaRN 1M-context tables, the concurrent and single-request
serving tables, the cross-engine comparison against vLLM, and the retrieval and soak results were all
measured on other hardware and other artifacts, mostly on the V100X2 and RTX 5090 forks. They are
inherited evidence. They do not establish 2080 Ti performance, and they do not update any figure on
this page. The V100X2 tables remain in [Performance](docs/performance.md); the RTX 5090 campaigns are
reduced there to a one-line-per-campaign summary, with their raw records under
[`eval/results/`](eval/results).

The sibling V100X2 fork carries its own README and the full V100 measurement programme; the
measurements that belong to this checkout are the acceptance table above, the
[concurrent decode](docs/performance.md#concurrent-decode-on-the-turing-target) table, the
[KV-cache dtype](docs/performance.md#kv-cache-dtype-on-the-turing-target) and its
[FP8-E4M3](docs/performance.md#fp8-e4m3-kv-cache-on-this-target) subsection,
[TP2 transport](docs/performance.md#tp2-transport-on-this-target), and the
[SM75 external reference](docs/performance.md#external-sm75-reference-vllm-2080ti-definitive)
comparison against `vLLM-2080Ti-Definitive`.

## DFlash2 speculative route

DFlash2 is an optional five-layer BF16 draft package for the registered
`qwen3.8-27b/groupwise-int` and `qwen3.8-27b/gguf-q4-k-m` artifacts. Include it during GGUF
conversion with `--dflash2-model`, or append it to an existing groupwise-int artifact with
`tools.convert.qwen3_8_27b.attach_dflash2`; all five draft layers use local attention, so this route
needs no growing DFlash Full KV pool. It includes dynamic grouped convolution, lattice selection,
TP2 replicated verification, and graph-stable selector scratch. Enable it with
`--spec dflash --draft-tokens 3` (up to seven), with or without `--tp 2`.

The recorded DFlash2 throughput comparisons come from a temporary llama.cpp build on two V100s and
the V100X2 artifact. They are **not** measurements of this target and are not reproduced here; see
[Performance](docs/performance.md). A short TP2 functional smoke passed on the local 2080 Ti pair,
but DFlash2 has not been performance-profiled or acceptance-tested there, and this checkout makes no
DFlash throughput claim.

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem). They are
**inherited from the upstream campaign** and were not re-measured on this target; they are recorded
because they characterise the artifacts, not this hardware:

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8-27B rows used
temperature 1.0 and presence penalty 0.0. The multimodal columns (ERQA and RealWorldQA) ran with
`--vision` at a 81,920-token context limit; the text columns used a 262,144-token limit except
Qwen3.8-27B NVFP4, which needs 252,928 to fit the RTX 5090 after weights.

These are single-sample results under that NInfer evaluation profile, not pass@k. See the model
cards and [full performance document](docs/performance.md) for correct/total counts and evaluation
notes.

## Requirements

NInfer currently requires:

- 64-bit Linux;
- two NVIDIA GPUs of the same compute capability for the TP2 profile; this checkout is qualified on
  two **RTX 2080 Ti 22 GB** cards (`sm_75`, Turing);
- NVIDIA driver support for the selected GPU and a CUDA toolkit for the target architecture; this
  checkout's current `sm_75` build tree uses CUDA 13.2, and CUDA 12.8 (not 13.x) remains required
  for the Volta `sm_70` sibling;
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config`;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja, when using the commands below.

The build accepts `70`, `75`, `86`, or `89`. `70` remains the CMake default for the sibling V100
checkout, so this target must pass `-DCMAKE_CUDA_ARCHITECTURES=75`. RTX 5090 `120a` is an upstream
configuration that this checkout's CMake rejects; on `sm_75` the NVFP4 A4 and FP8 A8 execution
leaves therefore compile to stubs that throw at run time, and the tests that exercise them fail
rather than skip (see [Current limits](#current-limits)). There is no install target and no
packaged binary distribution; NInfer runs from its source build tree.

## Build

The build tree used here is `build/`, configured for Turing:

```bash
export CPATH=/data/deps/root/usr/include/x86_64-linux-gnu \
       LIBRARY_PATH=/data/deps/root/usr/lib/x86_64-linux-gnu \
       PKG_CONFIG_PATH=/data/deps/root/usr/lib/x86_64-linux-gnu/pkgconfig

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j 10
```

The `export` line points at the private dependency prefix this machine uses. When the host lacks the
required FFmpeg or curl versions, `tools/v100/build_dependencies.sh` builds that prefix under
`build/_deps/install` instead.

The configuration builds:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

Tests, benchmarks and maintainer tools are excluded from the default build; see
[Contributing](CONTRIBUTING.md) for how to build and select them. The inherited `Dockerfile` uses
CUDA 13.1 and has not been retargeted for Turing; use the source build above. The `build/` tree on
this machine was configured with `/usr/local/cuda-13.2/bin/nvcc`.

## Convert a local GGUF instead

Both Qwen3.8-27B profiles are used here as published `.ninfer` artifacts, but the GGUF-derived
`gguf-q4-k-m` identity can be produced locally. Conversion takes the exact Q4_K_M GGUF and its BF16
companion vision file, writes a native `.ninfer` artifact plus a conversion report, and never
downloads a replacement checkpoint:

```bash
python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --out /Models/ninfer-2080Ti/qwen3_8_27b_q4_k_m.ninfer
```

Use a Python environment containing NumPy. The runtime accepts the resulting `.ninfer`, not the
source GGUF. This identity preserves the source Q4_K/Q6_K codes and embedded scales, supports Text
and MTP, and rejects Vision; see the
[GGUF artifact contract](docs/maintainer/qwen3.8-27b-artifact.md#14-preserved-gguf-q4_k_m-artifact).
It has not been used for the acceptance measurement above, which uses the published
`qwen3.8-27b/groupwise-int` artifact.

## Download a model

Use the Hugging Face CLI to download one of the registered artifacts:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# Or the 27B NVFP4 weight variant:
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

# Or Qwen3.8-27B:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

# Or Qwen3.8-27B NVFP4:
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

Current NInfer builds accept only the version-2 artifact container, and all five downloads above
are version 2. Migration applies only to Qwen3.6 artifacts downloaded before their version-2
publication; both Qwen3.8-27B profiles were published directly as version 2. Migrate an older exact
local file in place:

```bash
python3 -m tools.artifact.migrate_v1_to_v2 models/qwen3_6_27b.ninfer
```

Use the same command with `qwen3_6_27b_nvfp4.ninfer` or `qwen3_6_35b_a3b.ninfer` for those
artifacts. The migration updates only container metadata; it does not rewrite the weight payload.
Alternatively, download the current version-2 file again from its Hugging Face repository.

Each `.ninfer` file contains the weights and frontend resources needed by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

Each artifact is complete, while GPU residency is fixed at process startup. Speculative decoding is
disabled by default, so MTP/DFlash state and the optimized proposal head are not uploaded.
Vision is also disabled by default, so its weights, Vision scratch phase, and frozen
request-transient allocation are omitted. Add `--vision` to the CLI or server process that must
accept image or video input. Disabled capabilities cannot be enabled by a later request. DFlash is
text-only: Qwen3.8-27B groupwise-int or GGUF artifacts can include the optional DFlash2 package,
while Qwen3.6-35B-A3B uses its separate DFlash package.

## Run the CLI

`scripts/run.sh` wraps this machine's dual-GPU defaults -- TP2 on `--devices 0,1`,
`--max-context 262144` -- and forwards everything after the prompt verbatim, so any later scalar
flag overrides the injected default:

```bash
scripts/run.sh "Explain prefill and decode in three sentences."
scripts/run.sh --messages examples/cli/messages/text_smoke_zh.json --max-new 128 --greedy
NINFER_DEVICES=0,2 scripts/run.sh "..."               # the acceptance pair
NINFER_KV_DTYPE=fp8 scripts/run.sh "..."              # half footprint without the int8 accuracy loss
NINFER_KV_DTYPE=int8 scripts/run.sh "..."             # when KV capacity matters more than speed
```

The acceptance command from the table above, spelled out. Substitute any prompt that occupies about
85,000 tokens for `<prompt.json>`:

```bash
./build/apps/ninfer /data/models/qwen3.8-27b/qwen3_8_27b_v2.ninfer \
  --messages <prompt.json> \
  --max-new 64 --tp 2 --devices 0,2 \
  --max-context 262144 --kv-dtype fp16 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --greedy --raw-output
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr. See the [CLI guide](docs/cli.md), including
[choosing the KV-cache dtype](docs/cli.md#choosing-the-kv-cache-dtype), and the
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

`scripts/serve.sh` runs the server as a supervised background process with a PID file, an
unexpected-exit restart loop and `start|stop|restart|status|logs` subcommands:

```bash
scripts/serve.sh start
scripts/serve.sh status
scripts/serve.sh logs
scripts/serve.sh stop
```

It defaults to TP2 on `--devices 0,1` with `--max-context 262144`; `NINFER_HOST`, `NINFER_PORT`,
`NINFER_DEVICES`, `NINFER_KV_DTYPE`, `NINFER_API_KEY` and `NINFER_MODEL_ID` override those defaults.
The equivalent direct invocation is:

```bash
./build/apps/ninfer-serve /data/models/qwen3.8-27b/qwen3_8_27b_v2.ninfer \
  --host 127.0.0.1 --port 8000 --tp 2 --devices 0,2 \
  --max-context 262144 --kv-dtype fp16 --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Prefix caching is enabled by default on this target, including TP2 + MTP3. Send the conversation
history normally: compatible follow-up requests reuse the retained prefix and prefill only the new
suffix. An exact retained-frontier or saved-checkpoint hit can generate without re-prefilling prompt
tokens. The completion log reports `cache=` and `reuse=`; use `--no-prefix-reuse` to disable caching
for a cold comparison.

Reuse requires a complete saved model state at the resident frontier or a typed turn/response
checkpoint. Editing earlier history may require a full prefill; an arbitrary matching token prefix
is insufficient. The cache belongs to the running Engine and is lost on restart. With one active
slot, an unrelated request can replace the retained conversation.

Reuse saves prompt work, not decode time. The prefix-cache latency measurements in this repository
were taken on the V100 pair; **cached versus cold time to first token has not been timed on this
target**. See [cache validation and reproduction](docs/performance.md#v100x2-prefix-cache).

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages, token counting, and multimodal input. See
[HTTP serving](docs/serving.md).

## Dual-GPU (TP2), context capacity and YaRN

`--tp 2` splits one resident model across two devices and requires an explicit `--devices A,B`
naming two distinct devices of the same compute capability. TP2 runs one process, one resident model
and two CUDA devices without distributed serving, and is implemented for the 27B execution package
(`qwen3.6-27b` and `qwen3.8-27b`); `qwen3.6-35b-a3b` has no tensor-parallel path and rejects `--tp 2`
at startup.

This machine has three 2080 Ti cards, so the pair is a real choice, and it is settled by measurement
rather than by assuming that peer-to-peer always wins:

- **`--devices 0,1`** is the NVLink pair. Startup probes both directions, requires exact copies at
  every probed size, and then uses the direct route. Prefill is faster here (967 tok/s against 849 at
  85,000 tokens), but card 1 is the slowest of the three and it sets the decode round time.
- **`--devices 0,2`** has no peer access, so collectives run over the validated pinned host-staged
  route. Prefill gives up 12%, decode gains 4.7%, and acceptance is bit-identical.

Both are correct configurations; `0,2` is the acceptance pair because the objective here is decode.
`--tp 1` remains the default and is unchanged: `scripts/tp1-regression.sh` compares this build's
greedy output against token streams recorded from upstream `ninfer` at base commit `feaf4dd`, over a
short chat, a 2,191-token instruction and a 28,677-token document, and requires the token ids, the
generated text and the deterministic summary rows to be byte-equal. Its scope is exactly that: greedy
text decode, `--tp 1`, `--rope native`, NVFP4 weights, `qwen3.8-27b`, MTP off, concurrency 1. See
`tests/data/tp1-golden/MANIFEST.md`.

The registered native context ceiling is 262,144 tokens, and that is what this target is qualified
at. `--rope yarn` raises the ceiling to 1,048,576 tokens when `--yarn-origin` equals the artifact's
registered native capacity (`262144`) and `--yarn-factor` is a finite value in `[1.0, 64.0]` whose
product is a whole token count at or below `1,048,576`:

```bash
# YaRN is implemented, but 1M is a 32 GB-per-card configuration.
./build/apps/ninfer /data/models/qwen3.8-27b/qwen3_8_27b_v2.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-dtype int8 --kv-capacity auto \
  --messages long_prompt.json --max-new 256 --no-thinking
```

At 1M the KV pool alone was measured at about **17.7 GiB per device** with INT8 KV on the V100 pair,
which does not fit alongside the sharded weights on a 22 GB card. **The 1M configuration has not been
exercised on this target** and is documented here as an available option, not a qualified profile. At
262,144 tokens the three KV dtypes fit on the `groupwise-int` artifact; on the larger
`fp8-block128` artifact only the 8-bit pools reach that window, because its fp16 pool caps at
153,984 tokens (see [Acceptance on this target](#acceptance-on-this-target)). `--kv-capacity N`
sizes the shared Main Text pool explicitly,
`--kv-capacity auto` takes the largest usable capacity from the memory left after weights while
preserving 1 GiB of sizing headroom, and omission defaults to one `--max-context` worth of pages.

### Limitations

- **Vision is `--tp 1` only.** The Vision encoder runs on the primary device against replicated
  weights and has no split path, so `--tp 2 --vision` is rejected at startup. YaRN is likewise
  rejected together with `--vision`, because the encoder ropes 2-D image-grid positions.
- **Qwen3.8-27B DFlash2 supports TP2 for groupwise-int and GGUF identities when the optional
  companion package is present.** The Qwen3.6-35B-A3B DFlash route remains single-device because
  that target has no tensor-parallel path.
- **MTP is output-equivalent up to near-tie argmax flips, not bit-identical.** A verify round
  evaluates the target model over `K+1` columns at once and an ordinary round over one, which
  selects different GEMM shapes; greedy MTP-on and MTP-off streams can therefore diverge on a near-tie
  token. Every committed token is still one the target model's own argmax selected.
- **The NVFP4 A4 and FP8 A8 leaves are unavailable on `sm_75`.** They compile to stubs that throw,
  so NVFP4 W4A4 prefill and FP8 A8 execution are not supported paths on this target. The `nvfp4`
  artifact's A16 decode kernels and the separate block128 W8A16 path are unaffected.
- **`--ignore-eos` is a diagnostic flag.** It exists for fixed-length soak and throughput work.
  Generation past the end-of-turn token is off-distribution and is not a product output.
- The decode split policy was tuned at 262k on this target; it has not been swept elsewhere.

The design decisions behind these features -- the collective transport, the shard map, the YaRN
constants, and what each correctness gate actually proves -- are in
[Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md).

## Capabilities

The five published artifact profiles support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- startup-bounded small-scale concurrent serving with true batched decode;
- MTP speculative decoding with draft windows from one to five;
- FP16 (default), FP8-E4M3 and INT8 group-64 KV cache;
- model- and thinking-mode-aware official sampling defaults, with explicit greedy, temperature,
  top-k, top-p, min-p, and presence/frequency-penalty overrides;
- compatible-prefix reuse;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming and
  usage accounting;
- prompt-rendered function tools and parsed tool calls.

The Qwen3.8-27B groupwise-int and GGUF identities support text-only DFlash2 with one to seven draft
tokens when the optional package is embedded; both routes support TP1 and TP2. The 35B-A3B target
separately supports text-only DFlash with one to fifteen draft tokens at TP1.

The local `gguf-q4-k-m` identity supports text and MTP, including the public CLI and HTTP Engine
route. It rejects Vision. The acceptance workload on this target uses **one active request, native
RoPE, a 262,144-token capacity, an 85,070-token occupied prompt, and a maximum MTP draft window of
three** (`--kv-dtype fp16`, greedy, CUDA Graphs).

## Current limits

- Only the seven `(model_id, weights_id)` artifact identities listed above are registered.
  The acceptance measurements use `qwen3.8-27b/groupwise-int`; block-FP8 has the separately
  labeled research results above. Registration does not qualify every identity on this target.
- This checkout targets Volta, Turing, Ampere and early Ada. One CUDA device is the generic CLI
  default; `scripts/run.sh` selects exactly two with `--tp 2 --devices A,B`.
- One Engine owns one resident model and supports a startup-fixed capacity of 1–8 active requests.
  Decode-ready requests are compacted at round boundaries and executed in one batched model
  traversal.
- NInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  CPU/GPU offload, or distributed serving. Multi-GPU execution is exactly the two-device
  tensor-parallel width described above: one process, one resident model, no more than two devices.
  CUDA peer access is used wherever startup confirms it, including the `0,1` pair on this machine; a
  pair that does not advertise access, or whose copies fail the exact startup check — such as `0,2`
  here — uses the validated pinned host-staged transport.
- `--max-context` is the logical ceiling of each sequence and is configurable up to the registered
  models' native 262,144-token limit. The 1,048,576-token YaRN ceiling is implemented but is not a
  qualified profile on this target. `--kv-capacity N` explicitly sizes the shared Main Text KV pool
  for all active and retained sequences, while `--kv-capacity auto` selects the largest usable
  capacity from the memory remaining after weights are loaded while preserving 1 GiB of sizing
  headroom. Omission defaults to one `--max-context` worth of pages. The resolved pool is fixed at
  startup and is not divided statically among request lanes.
- **Historical full-suite snapshot, not a current qualification gate.** The NVFP4 A4 and FP8 A8 execution leaves are stubbed out
  for architectures other than `120a`, so the tests and benchmarks that exercise them abort with
  `NVFP4 A4 execution requires an sm_120a GPU` (or the FP8 A8 equivalent) instead of skipping, and
  the NVFP4-A4 legs of composite suites abort with them. These failures are architectural, not
  numerical. The earlier full-suite run reported
  80 passing, 28 failing and 14 not-run of 122; the 28 failures are the arch-stub cases, missing
  local resources, and pre-existing `int8`-g64 reduction and schema cases.
  It does not establish current block-FP8 route coverage; see the focused checks in
  [tests/README.md](tests/README.md#block-fp8-checks).
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [CLI examples](examples/cli/)

## License

NInfer is licensed under the [Apache License 2.0](LICENSE). This fork's modifications are under the
same licence; see [NOTICE](NOTICE) for the attribution required by Apache-2.0 §4(b).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.

## Acknowledgements

The design and measurements on this page were informed by the upstream
[Neroued/ninfer](https://github.com/Neroued/ninfer) project, the RTX 3060 TP2 work, and
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100), whose Turing and Volta
investigations this checkout builds on.

A separate debt is owed to **[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)**,
the SM75-focused vLLM fork for dual RTX 2080 Ti (and for Tesla T10/T40/T4, TITAN RTX and Quadro RTX
6000/8000). Two things came from studying it:

- **The external same-GPU-class reference.** Its published 2x2080Ti table was measured on a
  different host/toolchain and physical GPUs `1,5`, not locally on `0,1`. Its FP8-weight rows
  include both FP8 KV (1393.94/92.47 prefill/decode tok/s) and FP16 KV (1425.34/95.10), with
  MTP4. These are directional references, not matched speedups or evidence that FP8 KV explains
  the local gap. Marlin retains compressed FP8 weights and dequantizes register fragments in the
  GEMM loop; W8A16 does not mean a resident full-FP16 weight expansion. A local concurrency run of
  that stack on `0,1` -- FP8 KV, MTP4, one to three simultaneous requests -- is archived under
  [`eval/results/vllm-fp8-concurrency/`](eval/results/vllm-fp8-concurrency/README.md); it is a
  concurrency measurement, not a reproduction of the published rows.
- **The prefill accumulation form.** The `__CUDA_ARCH__ == 750` branches in the Marlin files that
  fork vendors (upstream vLLM/Marlin code, not the fork's own) set `use_fp16_accum = true` for the
  W8A16 route, so `m16n8k8.f16.f16.f16.f16` runs with a **pure fp16 accumulator over the whole K
  and no fp32 reduction across segments**. This checkout's block-FP8 prefill HMMA now matches that
  form: dropping the per-segment fp32 fold measured **+10.3% prefill** in same-session alternating
  A/B (swiglu leaf 44.9 to 54.7 TFLOPS), with the argmax gate bit-identical to the folded engine.
  Its vendored CUTLASS `m8n8k16.s8` dispatch independently
  corroborated the int8 tensor-core form used here. Credit for those kernels belongs to upstream
  vLLM, Marlin and CUTLASS; the fork's contribution is lowering the Turing gates on the Python side
  so the routes run at all on `sm_75`.

Both are recorded with the comparison table and the measured scope in
[External SM75 reference](docs/performance.md#external-sm75-reference-vllm-2080ti-definitive). Its
companion **[weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)** GDN
forward-only kernel is a concrete alternative to this checkout's chunked WY/UT formulation; it was
evaluated and recorded in the same section as not a lever at this scale.

The prefill investigation consulted [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)'s
SM70-oriented kernels and vLLM scheduling ideas. DFlash2 integration follows [Inco AI](https://inco.ai/blog/dflash2/) and its [Qwen3.8-27B draft checkpoint](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp)
was reviewed for RAM-backed KV ideas, but its selective-history attention semantics are not enabled
in this full-context product. The prefill and DFlash2 comparison methodology is documented
in [Serving performance](docs/performance.md).
