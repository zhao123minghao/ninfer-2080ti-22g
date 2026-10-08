# External reference: `vLLM 2080 Ti Definitive Edition`

Reference notes for the third-party SM75 serving stack that this checkout compares against.
Everything here describes **another project's runtime**, not NInfer. It exists so that a future
maintainer can read the external tables, profiles, and patches without re-cloning or
re-reverse-engineering them, and so that no external number is silently mixed with a local one.

Sources read for this note (2026-10-05):

- local checkout: `~/ws/vLLM-2080Ti-Definitive/` (release `0.2.2-post3`)
- upstream project: `github.com/weicj/vLLM-2080Ti-Definitive` (fork of `vllm-project/vllm`,
  Apache-2.0), companion project `2080Ti-LLM-Toolbox`
- published performance tables: `profiles/2x2080Ti/README.md`, `profiles/2xT10/README.md`,
  `profiles/4xT10/README.md` inside that checkout
- release history and pins: `CHANGELOG.md`, `VERSION`, `PROJECT_RELEASE.env`
- build and launch: `build.sh`, `update.sh`, `launcher.sh`, `tools/validate_profiles.sh`
- SM75 runtime paths: `vllm/v1/attention/backends/{flashinfer,fa_utils,turboquant_attn}.py`,
  `vllm/v1/attention/ops/triton_turboquant_decode.py`,
  `vllm/model_executor/layers/quantization/turboquant/{config,centroids}.py`,
  `vllm/model_executor/models/{qwen3_dflash,dflash2_sm75,qwen3_dflash2}.py`,
  `vllm/v1/spec_decode/dflash.py`, `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`,
  `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
  `tools/flashqla_sm75_patches/gdn_forward.cu`, `csrc/quickreduce/`, `vllm/config/cache.py`,
  `vllm/model_executor/layers/quantization/{fp8,modelopt}.py`
- tooling: `tools/profile_request.py`, `tools/{checkpoint_fingerprint,recommend_gpu_topology}.py`,
  `benchmarks/replayssm/e2e_decode_speedup.py`

Related local material: [performance.md](../performance.md#external-sm75-reference-vllm-2080ti-definitive)
carries the comparison table and the two items this checkout actually adopted from the stack
(the Marlin prefill accumulation form, and the Turing gates on the Python side). This document is
the wider map of the project; the performance page remains the authority for what *we* measured.

## 1. What the project is

A hardware-focused fork of vLLM that keeps SM75 (Turing) serving routes alive: RTX 2080 Ti 22 GB
pairs over NVLink, Tesla T10 (16 GiB, PCIe) pairs, and four-T10 TP4 nodes. It ships the runtime
patches, launch profiles, and validation evidence for those routes.

| Item | Value (from `VERSION`, `PROJECT_RELEASE.env`) |
|---|---|
| Fork release | `0.2.2-post3` (2026-10-01) |
| Upstream base | `b23433088b` = vLLM `v0.29.1rc0-33` |
| Runtime identity | `vllm-def-cu130` |
| Validated stack | CUDA 13.0, PyTorch 2.13.0+cu130, Python 3.12, Ubuntu 26.04+/kernel 7/GCC 15 |
| Legacy line | `v0.1.x` = CUDA 12.8 / PyTorch 2.11, explicitly unmaintained |
| Weight routes | Qwen3.8-27B FP8 (`qwen27b/w8a16`), Qwen3.8-27B NVFP4 (`w4a16`), Qwen3.8-27B INT4, Qwen3.6-35B-A3B FP8 (`qwen35b/w8a16`) |

`sm_75` is the only tensor-core generation this fork targets, so its patch surface and its
"what works on Turing" answers are the interesting part. Upstream vLLM remains the source of the
engine, scheduler, and most kernels; the fork's own contributions are guarded SM75 paths, launch
profiles, and validation.

Not to be confused with two other external sources this repo cites: the `inco.ai` DFlash2 blog
(Qwen3.8-27B drafter on V100 x2) and `vLLM-2080Ti-Definitive`'s own tables below. They are
different hosts and different stacks.

### 1.1 Release lineage

The `pre` / `pre2` / `pre3` / `pre4` / `RC1` / `0.2.1` / `0.2.2` / `post1..post3` sequence is a
CUDA 13 migration built on upstream nightlies. `pre` moved the tree to CUDA 13.0 / torch 2.13 /
Python 3.12 on upstream `v0.27.1`; `pre4` re-based on `b23433088b` (`v0.29.1rc0-33`) and states
that from then on the fork "retains only the SM75-specific FlashQLA, TurboQuant, parser, and
rendezvous paths". The `post1..post3` patches are hybrid-Mamba/DFlash2 correctness work
(prefix-cache replay, target speculative state pages, Mamba align accounting) plus launcher
interaction and the opt-in SSD prefix cache.

### 1.2 What the fork itself owns

Reading `CHANGELOG.md` against the tree, the fork-owned parts of the SM75 route are:

- build and runtime glue: CUDA 13 pins, mirror handling, `uv` installs, cache isolation, the
  launcher and its profiles;
- the FlashQLA SM70/SM75 import patch, the hand-written SIMT GDN kernel, and its loader;
- the TurboQuant backend integration on SM75, including the FP8-key format choice;
- DFlash/DFlash2 hosting on Model Runner V2, the SM75 BF16 transport codec, and the
  speculative FA2 graph wrappers;
- Turing quant dispatch (Marlin W4A16 and MarlinFP8 gates, MoE fallbacks);
- custom all-reduce qualification over PCIe links.

Kernels such as Marlin and the vendored CUTLASS sources are upstream code; for those the fork
mainly lowers the Turing gates so the routes run at all.

## 2. Local checkout state on this host

Recorded so that "did we already run it?" has an answer:

- `.venv` exists and reports **Python 3.12.13 / torch 2.13.0+cu130 / CUDA 13.0**; `build-logs/`
  holds several `build-*.log` entries from 2026-10-03.
- `run-logs/vllm-Qwen3.8-27B-FP8-20261003-134615.log` is a completed service start (13:51-15:23)
  of the FP8 route on this host. Its resolved command line is the concrete example in section
  5.3 below.
- That log ran with `--disable-log-stats`, so it contains no throughput measurement. The only local
  measurement of this stack is the 2026-10-07 concurrency run in section 8.3; every number in the
  profile tables is the external validation host's (section 8).
- `run-logs/start-manager.state`, `model-target-history`, `model-draft-history` are launcher state
  files, not evidence.

### 2.1 What that run log resolved (engine configuration)

The log prints the engine's final configuration, which is the cheapest way to see how the stack
actually wires up on this hardware:

- engine: V1 LLM engine `v0.2.2.post3`, `dtype=torch.float16`, TP2, prefix caching on, chunked
  prefill on, `kv_cache_dtype=fp8`, `mamba_cache_mode=align`, `max_seq_len=186368`,
  `SpeculativeConfig(method='mtp', num_spec_tokens=4)`;
- compile config: `custom_ops=['+quant_fp8','none','+quant_fp8']`, splitting ops include
  `vllm::unified_attention_with_output`, `vllm::qwen_gdn_attention_core`, `vllm::short_conv`,
  `vllm::mamba_mixer2`; `cudagraph_mode=FULL_AND_PIECEWISE`, `cudagraph_capture_sizes=[5]`,
  `max_cudagraph_capture_size=5`; inductor passes `fuse_norm_quant=True`, `fuse_act_quant=True`,
  `fuse_attn_quant=False`;
- attention selection: `Using FLASHINFER attention backend out of potential backends:
  ['FLASHINFER', 'TRITON_ATTN']`;
- GDN: `Using FlashQLA legacy SM70/SM75 GDN prefill kernel (requested=flashqla_legacy,
  head_k_dim=128)`;
- spec graph: `Created SM75 speculative FA2 graph wrapper: batch=1, query=5, heads=12/2,
  head_dim=256, page_size=16, causal=True` (K=4 -> query width 5 = 1 bonus + 4 drafts);
- capacity: `GPU KV cache size: 190,540 tokens, Maximum concurrency for 186,368 tokens per
  request: 1.02x`;
- loader notes: the draft's max model len is clamped from 262144 to the target's 186368, and the
  scheduler warns that `max_num_scheduled_tokens` was reduced to `MAX_BATCHED_TOKENS=2048` for
  the speculative route ("may lead to suboptimal performance");
- upstream's own warning is printed: `num_speculative_tokens > 1` runs multiple forwards through
  the same MTP layer and may lower acceptance.

Because the run had `--disable-log-stats`, none of this produced throughput numbers.

## 3. Build pipeline (`build.sh`, `update.sh`)

`build.sh` is a linear script that logs to `build-logs/build-<stamp>.log` and fails loudly:

1. **Environment.** Source `PROJECT_RELEASE.env`; pick `CUDA_HOME` from `/usr/local/cuda-13.0` or
   `/usr/local/cuda-13`; put `uv` on `PATH`; reuse prepared source trees under `.deps/`
   (`triton_kernels`, `cutlass`) so a rebuild does not re-clone them; accept a `git` mirror prefix.
2. **Torch and build tools.** `install_torch_from_mirror()` installs the pinned torch wheel;
   `install_vllm_build_tools()` adds the build-time toolchain. Mirrors are used when configured.
3. **FlashQLA SM70/SM75 backend** (`prepare_flashqla_sm75()`, gated by `FLASHQLA_ENABLED`). This is
   the fork's main *build-time* deviation:
   - clone `weicj/FlashQLA-SM70-SM75` (depth 1) into `.deps/FlashQLA-SM70-SM75`;
   - rewrite its package `__init__` files through `tools/patch_flashqla_sm75_imports.py` so that
     SM90-only TileLang imports degrade to `None` instead of raising;
   - overlay `tools/flashqla_sm75_patches/` (notably a hand-written `gdn_forward.cu`, section 7.4);
   - `uv pip install --no-deps -e <flashqla_dir>`, then JIT-build the "legacy" extension into
     `TORCH_EXTENSIONS_DIR` for `TORCH_CUDA_ARCH_LIST=7.5`.
4. **vLLM itself.** `uv pip install --no-build-isolation -e . --torch-backend=<cuda_backend>`,
   i.e. a source build against the installed torch; CUTLASS/triton/flashinfer come from
   `.deps/`, cache dirs, or `FLASHINFER_WORKSPACE_BASE` (FlashInfer AOT is enabled by default).
5. **Runtime check** before printing `BUILD SUCCEEDED`.

`update.sh` refreshes an existing checkout to the newest GitHub release while preserving local
environments, dependency caches, logs, results, and `profiles/local`, then offers to rebuild.

Runtime pins visible in this checkout:

| Item | Pin |
|---|---|
| CUDA / torch | CUDA 13.0, `torch 2.13.0+cu130` (the upstream nightly snapshot decides this; the fork follows it) |
| Python / OS | 3.12, Ubuntu 26.04+, kernel 7.x, GCC 15 (`gcc-12`/`gcc-14` are also probed for host code) |
| INT6 / AutoRound | `humming-kernels[cu13]==0.1.13` (per the changelog) |
| FlashQLA | `weicj/FlashQLA-SM70-SM75`, depth-1 clone under `.deps/` |
| CUTLASS / Triton kernels | prepared source trees under `.deps/` (`cutlass-*`, `triton_kernels-*`) |
| FlashInfer | AOT build (`FLASHINFER_ENABLE_AOT=1`), workspace isolated under the runtime tree |
| Host compilers | the launcher exports `CUDA_HOME`, `CUDA_PATH`, `CUDACXX`, `CUDAHOSTCXX` and picks `CC`/`CXX` from `gcc-12`/`gcc-14` when unset |

Their `AGENTS.md` is the project's own discipline document: keep upstream attribution, prefer
small guarded patches over rewrites, treat `profiles/<hardware>/README.md` as the source of truth
for shipped profiles, and require capacity **and** throughput evidence before promoting a route.
Validation commands they expect: `bash -n build.sh launcher.sh tools/validate_profiles.sh`,
`bash tools/validate_profiles.sh`, `python3 -m py_compile <files>`, `git diff --check`, and
`launcher.sh --print-config` for any launcher/profile change.

## 4. Profiles and modes (`profiles/`, `tools/validate_profiles.sh`)

Layout is flat: `profiles/<hardware>/<model>/<weight>/<route>.env`, filename
`<decoder>-<kv>-<concurrency><context>-<message>.env`, e.g.
`dflash2-fp8kv-1x262K-text-image.env` = DFlash2 decoder, FP8 KV, one request, 262K decimal-token
context, text+image messages.

Profiles carry **route parameters only** (per their rule): `MODEL_FAMILY`, `MODEL_VARIANT`,
`QUANTIZATION`, `KV_CACHE_DTYPE`, `MAX_MODEL_LEN`, `GPU_UTIL`, `MAX_NUM_SEQS`, `MESSAGE_TYPE`,
`SPECULATIVE_METHOD` (`none|mtp|dflash`) and `SPECULATIVE_TOKENS` (defaults `0`, `3`, `7`), plus
optional `MODE`, `MAX_BATCHED_TOKENS`, `ENABLE_YARN`. GPU selection, port, chat template, and
reasoning defaults stay in the launcher. Shipped 2x2080Ti routes:

| Route file (under `profiles/2x2080Ti/`) | KV | Spec | Seq | Context |
|---|---|---|---|---|
| `qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env` | FP16 | MTP/4 | 1 | 148K |
| `qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env` | FP8 | MTP/4 | 1 | 262K |
| `qwen27b/w8a16/mtp4-fp8kv-1x186K-text-image.env` | FP8 | MTP/4 | 1 | 186K |
| `qwen27b/w8a16/nomtp-fp16kv-1x176K-text-only.env` | FP16 | none | 1 | 176K |
| `qwen27b/w8a16/nomtp-fp16kv-1x121K-text-image.env` | FP16 | none | 1 | 121K |
| `qwen27b/w8a16/yarn-fp8kv-1x338K-text-only.env` | FP8 | none + YaRN | 1 | 338K |
| `qwen27b/w4a16/dflash2-fp8kv-1x262K-text-image.env` | FP8 | DFlash2/7 | 1 | 262K |
| `qwen27b/w4a16/dflash-fp8kv-2x176K-text-only.env` | FP8 | DFlash/7 | 2 | 2x176K |
| `qwen27b/w4a16/mtp4-fp8kv-2x229K-text-only.env` | FP8 | MTP/4 | 2 | 2x229K |
| `qwen27b/w4a16/mtp4-tq4nc-3x262K-text-only.env` | TurboQuant 4-bit NC | MTP/4 | 3 | 3x262K |
| `qwen27b/w4a16/yarn-fp8kv-1x524K-text-only.env` | FP8 | none + YaRN | 1 | 524K |
| `qwen35b/w8a16/nomtp-fp16kv-1x262K-text-only.env` | FP16 | none | 1 | 262K |
| `qwen35b/w8a16/nomtp-fp8kv-1x221K-text-image.env` | FP8 | none | 1 | 221K |

Modes are launcher-selected and default to `fast`: `safe` (troubleshooting; PIECEWISE graphs only),
`normal` (stable daily), `fast`, `aggressive` (highest performance, extra quality risk).

The full shipped matrix is 27 profiles across three platforms: 13 on `2x2080Ti`, 4 on `2xT10`
(INT4 W4A16 MTP3 routes), and 10 on `4xT10` (FP8 and NVFP4, DFlash2/MTP4, FP16/FP8/TQK8V4 KV).

`tools/validate_profiles.sh` is a real schema gate rather than a style checker:

- only 13 keys are allowed (`MODE`, `MODEL_FAMILY`, `MODEL_VARIANT`, `QUANTIZATION`,
  `KV_CACHE_DTYPE`, `MAX_MODEL_LEN`, `GPU_UTIL`, `MAX_BATCHED_TOKENS`, `MAX_NUM_SEQS`,
  `MESSAGE_TYPE`, `SPECULATIVE_METHOD`, `SPECULATIVE_TOKENS`, `ENABLE_YARN`), the required ones
  must be non-empty, duplicate keys are rejected, and no directory may be named `fast/` or
  `normal/` (the mode is chosen by the launcher, never stored in the path);
- `MODE` may be omitted or `fast`/`normal`; `safe`/`aggressive` stay launcher-only choices;
- `KV_CACHE_DTYPE` must name the stored precision explicitly -- one of `float16`, `fp8`,
  `turboquant_4bit_nc`, `turboquant_k8v4`; `auto`/`default`/empty are rejected;
- filename and fields must agree: `mtp<N>-` vs `SPECULATIVE_TOKENS`, `nomtp-` requires method
  `none`, `dflash*` requires method `dflash`, `-<N>x<ctx>K-` vs `MAX_NUM_SEQS` and
  `MAX_MODEL_LEN` (decimal thousands), `fp16kv`/`fp8kv`/`tq4nc`/`tqk8v4` vs the KV dtype,
  `text-only`/`text-image` vs `MESSAGE_TYPE`, and `yarn-` vs `ENABLE_YARN=1` (with the reverse:
  `ENABLE_YARN=1` requires a `yarn-` filename);
- `GPU_UTIL` strictly inside (0,1), positive integers for lengths/sequence counts, and the
  official profile set must match the expected list exactly (count and sorted order).

## 5. Launch pipeline (`launcher.sh`, ~5.7k lines)

### 5.1 Interactive flow

The menu is an eight-step flow (launcher banner comment): pick target (and optional draft)
checkpoint, pick a profile, configure GPU devices and TP/PP topology, pick the mode and network
options, start the service, then wait for `/health` and run a small smoke request through the
OpenAI API; a separate entry stops a running service. State lives in `run-logs/`.

### 5.2 Non-interactive use

```bash
MODEL_DIR=/path/to/checkpoint \
PROFILE=2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env \
MODE=fast GPU_DEVICES=1,5 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
./launcher.sh --print-config        # resolve a route without starting it
```

`docs/non-interactive-launch.md` documents automation; `tools/validate_profiles.sh` checks that
filenames and fields agree.

### 5.3 Server command (what the launcher actually runs)

`build_args()` assembles a `vllm.entrypoints.openai.api_server` invocation. The local 2026-10-03
run on this host resolved to:

```text
--host 127.0.0.1 --port 8000 --model /data/models/qwen3.8-27b/Qwen3.8-27B-FP8/
--served-model-name Qwen3.8-27B-FP8 --dtype half --tensor-parallel-size 2
--generation-config auto --gpu-memory-utilization 0.96 --max-model-len 186368
--enable-chunked-prefill --max-num-seqs 1 --max-num-batched-tokens 2048
--quantization fp8 --kv-cache-dtype fp8 --mamba-cache-mode align
--enable-prefix-caching --enable-prompt-tokens-details --disable-log-stats
--reasoning-parser qwen3 --limit-mm-per-prompt {"image":64,"video":0,"audio":0}
--additional-config {"gdn_prefill_backend":"flashqla_legacy"}
--speculative-config {"method":"mtp","num_speculative_tokens":4}
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE",
                      "cudagraph_capture_sizes":[5],"max_cudagraph_capture_size":5}
```

Notable pieces of that pipeline:

- `--dtype half` on Turing (FP16 activations for every route -- Turing has no BF16 tensor cores
  -- so weight precision is entirely the quantization method's choice), `--generation-config auto`
  (load the model's own generation config), `--enable-chunked-prefill`, and explicit
  `--max-num-seqs` / `--max-num-batched-tokens` per route.
- `--mamba-cache-mode align` for the Qwen hybrid GDN models; `--disable-hybrid-kv-cache-manager`
  exists as an escape hatch.
- `--additional-config {"gdn_prefill_backend":"flashqla_legacy"}` selects the SM75 GDN prefill
  kernel.
- Speculative config comes from the profile through `build_generated_speculative_config`. The
  generated JSON always carries `method` and `num_speculative_tokens`; for `dflash` it adds
  `model` (draft checkpoint), `draft_tensor_parallel_size`, `max_model_len`, `attention_backend`
  (their default for DFlash is `TRITON_ATTN`), `kv_cache_dtype`, `draft_sample_method` (`greedy`
  or `probabilistic`), `disable_padded_drafter_batch`, plus `use_local_argmax_reduction` when
  requested. A raw `SPECULATIVE_CONFIG` JSON overrides the generator, and
  `--per-request-spec-decode-metrics` adds per-request speculation statistics.
- CUDA-graph policy follows the mode: `safe` -> PIECEWISE; `normal` -> PIECEWISE with speculation,
  else FULL_AND_PIECEWISE; `fast`/`aggressive` -> FULL_AND_PIECEWISE. Without speculation the
  capture list is `[1]`; with speculation it is derived from `query_width = spec_tokens + 1`:
  `cudagraph_capture_sizes = [query_width * i for i in 1..MAX_NUM_SEQS]` and
  `max_cudagraph_capture_size = MAX_NUM_SEQS * query_width`. The FlashInfer SM75 speculative
  wrapper checks for exactly that set before it enables captured graphs (section 7.5).
- Optional: `--kv-transfer-config` (KV disk cache), `--attention-backend`, custom all-reduce
  on/off, `--long-prefill-token-threshold`, `--prefill-batch-barrier`, tool/reasoning parsers,
  chat-template file, `--language-model-only`, `--skip-mm-profiling`.

### 5.4 Turing runtime environment the launcher exports

| Variable | Value / purpose |
|---|---|
| `TORCH_CUDA_ARCH_LIST` | `7.5` |
| `VLLM_DISABLE_TILELANG` | `1` — TileLang's TVM-FFI init is incompatible with SM75 TP workers |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` for DFlash routes; the V2 runner hosts the DFlash/DFlash2 candidate selector, KV precompute, and graph manager |
| `VLLM_SM75_SPEC_SYNC_MODE`, `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH` | MTP graph policy for the SM75 route |
| `VLLM_SM75_FLA_NATIVE_NORM` | escape hatch: the FLA Triton RMSNorm JIT can deadlock four concurrent SM75 ranks |
| `FLASHQLA_ROOT` | checkout of `FlashQLA-SM70-SM75`; also put on `PYTHONPATH` |
| `TORCH_EXTENSIONS_DIR` | holds the JIT-built GDN legacy extension |
| `FLASHINFER_ENABLE_AOT`, `FLASHINFER_WORKSPACE_BASE` | FlashInfer AOT, isolated per worktree |
| `VLLM_INT8KV_FA_*` | int8 KV FlashAttention prefill/continuation/cascade dequant knobs, plus `VLLM_INT8KV_FA_CASCADE_TILE_TOKENS=65536`; **no consumer exists in this checkout's source tree or installed binaries, and no shipped profile selects `int8_per_token_head`** -- treat these as dormant |
| `VLLM_TURBOQUANT_*` | prefill via FlashInfer fa2, continuation workspace reserve (`65536`, or `262144` for a long single-sequence route), `CUDAGRAPH_SPEC_DECODE_SAFE=1`, `SPEC_DECODE_CHUNK_SIZE=1`, `CONTINUATION_SDPA_Q_CHUNK=512`, `CONTINUATION_SDPA_MAX_QK_CELLS=16777216`, `SPEC_CONTINUATION_DECODE_FASTPATH=1` |
| `VLLM_TURBOQUANT_DECODE_BLOCK_KV` | TQ decode tile (default `2`; one of 1/2/4/8/16); section 7.3 |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | kept inside the runtime tree |
| `PYTHONSAFEPATH`, `PYTHONUNBUFFERED` | set for the served process; sitecustomize hooks may print startup diagnostics |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING`, `VLLM_DEFAULT_THINKING_TOKEN_BUDGET` | set only when auto tool choice / a reasoning budget is configured |

Two of these are derived rather than fixed. The TurboQuant continuation workspace reserve
defaults to 65536 tokens and is raised to 262144 when `MAX_MODEL_LEN >= 240000` with
`MAX_NUM_SEQS <= 1` and prefix caching on, because the first long continuation would otherwise
grow the workspace at runtime and can trip OOM or a bad split path.

### 5.5 Health check and smoke request

After the API server starts, the launcher polls `GET /health` quietly until it answers, then runs
a smoke test: `GET /v1/models` to learn the served model id, then one chat completion with
`"Reply with OK."`, `max_tokens=64`, `temperature=0`, non-streaming, and
`chat_template_kwargs={"enable_thinking": false}`. Acceptance is textual -- it takes `content` or
`reasoning_content`/`reasoning` and fails on an empty message -- so the smoke test proves the
full request path works but says nothing about quality. The launcher's startup summary also
prints a "CUDA graph captures" line that reflects the same capture-size derivation
(`custom` when the user supplied `COMPILATION_CONFIG_JSON`).

## 6. Runtime feature map (where to look)

| Area | Path in the external checkout |
|---|---|
| Engine / model runner | `vllm/v1/...`, V2 runner under `vllm/v1/worker/gpu/` |
| Attention backends | `vllm/v1/attention/backends/{flashinfer,flash_attn,turboquant_attn,gdn_attn,rocm_attn}.py` |
| TurboQuant kernels | `vllm/v1/attention/ops/turboquant_soa/triton_turboquant_{store,decode,decode_v2,unified_attention}.py`, `vllm/v1/attention/ops/triton_turboquant_decode.py` |
| TurboQuant quantizer | `vllm/model_executor/layers/quantization/turboquant/{config,centroids}.py` |
| Speculative decoding | `vllm/v1/spec_decode/dflash.py` (proposer) and `.../utils.py` (fused input expansion); V2 runner: `vllm/v1/worker/gpu/spec_decode/{mtp/speculator,dflash2/speculator,autoregressive/{speculator,cudagraph_utils},dspark,eagle,adaptive_verification,rejection_sampler}` |
| DFlash draft models | `vllm/model_executor/models/qwen3_dflash.py` (attention/decoder/model base), `qwen3_dflash2.py` (grouped conv + candidate selector), `dflash2_sm75.py` (BF16 transport codec) |
| MTP draft models | `vllm/model_executor/models/qwen3_5_mtp.py` and many `*_mtp.py` families |
| GDN / hybrid Mamba | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `vllm/v1/attention/backends/gdn_attn.py` |
| KV cache / dtypes | `vllm/config/cache.py` (`CacheDType`), `vllm/v1/kv_cache_interface.py` |
| Quantization | `vllm/model_executor/layers/quantization/{fp8,compressed_tensors,modelopt,humming,moe_wna16,...}` |
| Marlin kernels | `csrc/libtorch_stable/quantization/marlin`, `.../moe/marlin_moe_wna16` |
| Custom all-reduce | `csrc/quickreduce/{base,quick_reduce,quick_reduce_impl}` |
| SM75 GDN kernel | `tools/flashqla_sm75_patches/gdn_forward.cu`, `tools/flashqla_sm75_patches/sm_legacy.py` |
| Build helpers | `build.sh`, `tools/{flashinfer-build.sh,patch_flashqla_sm75_imports.py,install_*}`, `rust/` |
| Ops tooling | `tools/{profile_request.py,checkpoint_fingerprint.py,recommend_gpu_topology.py,benchmark_flashqla_qkv_storage.py}`, `benchmarks/replayssm/e2e_decode_speedup.py` |

## 7. Algorithms

### 7.1 MTP (`SPECULATIVE_METHOD=mtp`, tokens 3 or 4)

Upstream vLLM's MTP path driven by the checkpoint's own MTP layer (`qwen3_5_mtp.py`), run
**sequentially**: each draft token needs another forward through the same MTP layer, which their own
startup warning states ("`num_speculative_tokens > 1` will run multiple times of forward on same MTP
layer, which may result in lower acceptance rate"). That is the same structure this checkout's MTP3
uses, and the same reason acceptance decays with draft width.

On Model Runner V2 the host class is `MTPSpeculator(AutoRegressiveSpeculator)`: it loads the MTP
layer through `load_eagle_model`, keeps one `SpeculatorCudaGraphManager` for prefill and one for
decode, and can run multi-step decode as a single fused graph (`use_fused_multi_step_decode`).
One checkpoint-driven optimization exists: when the draft config sets
`index_share_for_mtp_iteration`, step 0 computes the draft's top-k indices and later steps reuse
them (`set_skip_topk(True)` plus `compact_topk_indices` over each request's last token), with the
flag cleared at every prefill boundary so a failed step cannot leave reuse mode on.

Two scheduler-level facts show up in their runtime: `num_speculative_tokens > 1` repeatedly runs
the same MTP layer (with MTP-4 that is four draft forwards per step), and the route's
`max_num_scheduled_tokens` is reduced to `MAX_BATCHED_TOKENS` (their profiles use 2048), which
upstream warns may be suboptimal for speculative batches. Structurally, this is why the MTP rows
decode far slower than the DFlash2 rows: DFlash2 replaces the per-draft forwards with one
cross-attention pass (section 7.2).

### 7.2 DFlash / DFlash2 (`SPECULATIVE_METHOD=dflash`, tokens 7)

A **separate draft checkpoint** (`incoai/Qwen3.8-27B-DFlash2`) that is a *cross-attention*
drafter, not an MTP head. `DFlashProposer` (`vllm/v1/spec_decode/dflash.py`) sets
`pass_hidden_states_to_model=True` and sizes its query space as
`max_query_tokens = max_batch_size * (1 + num_speculative_tokens)`: the target's last hidden
states become the draft's **context K/V**, and the draft's queries are the bonus token plus mask
tokens -- `1 + K` rows produced in **one** forward. Mechanics worth recording:

- One fused Triton kernel (`_copy_and_expand_dflash_inputs` in `vllm/v1/spec_decode/utils.py`,
  registered with a warm-up configuration) expands the target batch into input ids, context and
  query positions, slot mappings, and the sample indices.
- Context preprocessing (`precompute_and_store_context_kv`) runs outside CUDA graphs: context
  states/positions are written straight into the draft's KV cache, while query buffers keep
  stable addresses for replay (`_context_slot_mapping_buffer` vs `_slot_mapping_buffer`, and a
  separate `_context_positions_buffer`). `dummy_run` therefore forwards only query tokens.
- Causality is per checkpoint: `dflash_has_any_non_causal()` decides whether the draft runs
  non-causal, and the metadata builder asserts non-causal support on every layer when it does
  -- the practical reason a DFlash route needs an attention backend with non-causal attention.
- The draft is **text-only**: the draft config clears the target's `is_mm_prefix_lm` flag while
  the target keeps image understanding (their profile note 4).

DFlash2 adds three draft-only mechanisms (`vllm/model_executor/models/qwen3_dflash2.py`):

- **Grouped convolution.** Each layer owns an `attention_conv` and an `mlp_conv`; both keep a
  base kernel `[2, taps, hidden]` and a `kernel_projection` that predicts, per token,
  `2 * taps * (hidden / group_size)` deltas. A token's effective coefficient for tap `t` is
  `base[t, channel] + delta[group(t), t]`; `prepare()` convolves the layer input and `finish()`
  the layer output (one Triton kernel with `TAPS`/`GROUP_SIZE` constexpr, 4 warps, element
  blocks of 1024/512). It is a short causal FIR whose coefficients are predicted from the
  current hidden state, applied on both sides of attention and of the MLP.
- **Candidate selector.** `CandidateSelector` holds predecessor/successor codebooks of shape
  `[vocab, selector_rank]` plus a hidden-size-to-rank projection. Candidate scoring is
  `unary_logits + <predecessor[candidate], hidden> * successor[candidate]`, computed as an
  einsum over rank with the anchor token as the first predecessor -- a learned edge score on
  top of the LM head's top-k (`selector_top_k`), exposed as `compute_candidates()` and wired
  through a dedicated logits processor that also applies the draft's `output_multiplier` and
  `final_logit_softcapping`. Draft embeddings are scaled by `input_embedding_scale`.
- **BF16-trained drafts on FP16 Turing hardware** -- see 7.2.1.

#### 7.2.1 SM75 BF16 transport codec (`dflash2_sm75.py`)

The DFlash2 checkpoint has BF16 activation boundaries while the Turing worker runs FP16. The
codec makes that cross-dtype contract explicit instead of hoping FP16 rounding is equivalent:

- constants: `_ACTIVATION_GAIN=32.0` (gate/up activations are pre-scaled into a BF16-friendly
  range), `_RESIDUAL_GAIN=256.0` (the residual stream is carried scaled in FP16 between
  sublayers), `_HALF_PAYLOAD_LIMIT=32752.0` (largest safe FP16 payload);
- BF16 rounding is emulated inside Triton (`_bf16_rne` does round-to-nearest-even on the FP32
  register with the tie bit), so the FP16 worker reproduces the training-time boundaries;
- each SwiGLU row is row-scaled by `exp2(ceil(log2(max(amax / 32752, L2 * down_bound / 32752))))`
  before FP16 storage; `down_projection_l2_bound` is the maximum L2 norm of one local output
  row of the following row-parallel projection, so Cauchy-Schwarz bounds the FP16 partial
  accumulation;
- tensor-parallel safety: `synchronize_dflash2_mlp_scales` all-gathers the per-row scales,
  takes the elementwise max across ranks, and rescales each rank's payload so every partial sum
  lives in one scale domain before the row-parallel reduction (the changelog's "one scale
  domain" fix);
- the residual state itself stays FP32 (`Sm75DFlash2ResidualNorm` combines
  `state * 256 + residual` in FP32, then normalizes with BF16-rounded weights);
- selection is `should_enable_dflash2_sm75`: FP16 runtime + BF16 checkpoint + capability
  exactly (7,5), opt-out via `VLLM_DFLASH2_SM75_TRANSPORT=0`.

This is the single most instructive file in the fork for "how to run a BF16-trained draft on
Turing without silently changing its numerics".

### 7.3 TurboQuant KV cache

KV compression with a data-free quantizer. The key path (`turboquant/centroids.py`,
`turboquant_attn.py`) is:

1. Rotate the key vector by an orthonormal **Hadamard matrix** (Sylvester construction, normalized
   by `1/sqrt(d)`, cached per dimension). The matrix is symmetric (`H = H^T`), so the inverse is
   the same matrix; no random sign flips are added because the file observes they cannot help a
   symmetric quantizer -- mirror centroids have identical distortion.
2. Each rotated coordinate is approximately `N(0, 1/d)` for `d >= 64`; the quantizer solves the
   Lloyd-Max conditions for that distribution (200 iterations, tolerance 1e-10, trapezoidal
   integration over `+/- 3.5 sigma`), yielding `2^bits` centroids and their midpoints, cached per
   (dimension, bits).
3. Keys are stored as centroid indices plus the per-vector norm in FP16. Decode inverse-rotates
   back (the continuation path uses an FP16 matmul copy of H to halve that bandwidth).
4. **Norm correction** (the `_nc` presets): centroid vectors are re-normalized to unit length
   before the inverse rotation, fixing quantization-induced norm distortion (~+0.8% PPL at
   4-bit). Values are uniform-quantized with per-vector FP16 scale and zero point.

Presets and layout:

| `--kv-cache-dtype` | keys | values | NC | compression | PPL delta |
|---|---|---|---:|---:|---:|
| `turboquant_k8v4` | FP8 (no rotation) | 4-bit | no | 2.6x | +1.17% |
| `turboquant_4bit_nc` | 4-bit MSE | 4-bit | yes | 3.8x | +2.71% |
| `turboquant_k3v4_nc` | 3-bit MSE | 4-bit | yes | ~3.5x | +10.63% |
| `turboquant_3bit_nc` | 3-bit MSE | 3-bit | yes | 4.9x | +20.59% |

- Packed bytes per head per position: key = `ceil(d * key_bits / 8) + 2` (the +2 is the FP16
  vector norm), except FP8 keys which are exactly `d` bytes; value =
  `ceil(d * value_bits / 8) + 4` (FP16 scale + zero). The slot is `[key | value]` rounded up to
  an even size so `effective_head_size = slot / 2` stays integral.
- **Boundary protection.** Dense models skip compression on the first and last `n=2` attention
  layers (capped at `num_layers/2`); the comment records ~30 GSM8K points lost on Qwen3-4B with
  aggressive presets otherwise. Hybrid models disable the protection because a hard n=2 on each
  side would cover ~40% of their 8-12 full-attention layers; the code logs the full-attention
  layer indices instead.
- Decode is a two-stage Triton path: stage 1 scores compressed keys and accumulates values per
  KV split (16-token tiles by default), stage 2 combines splits with log-sum-exp. Prefill runs
  ordinary SDPA on uncompressed K/V and quantizes on store.
- SM75 specifics: the decode tile defaults to `BLOCK_KV=2` via `VLLM_TURBOQUANT_DECODE_BLOCK_KV`
  (1/2/4/8/16 allowed) because that is the validated Turing register/shared-memory balance; FP8
  keys are encoded as **e5** (`_fp8_format_code`) because Qwen stores post-RoPE raw keys whose
  dynamic range on SM75 needs the wider exponent -- interpreting them as e4b15 loses retrieval
  quality and collapses later MTP acceptance (`e4nv` is rejected below SM89); prefill and
  continuation prefer a shared FlashInfer fa2 wrapper; speculative prefix reuse decodes the
  fixed-width MTP candidates as individual B=1 TQ decodes instead of one B=4 page-table decode
  (`VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1`), still fully captured -- only the graph
  topology differs.
- The fork lists its antecedents (DRIVE/EDEN, the HIGGS scalar case, "Cache Me If You Must") and
  states that QJL is deliberately omitted: independent groups found it amplifies variance through
  softmax.

### 7.4 GDN / FlashQLA on SM75

The Qwen hybrid layers need a chunked gated-delta-rule prefill. Upstream FlashQLA is TileLang/SM90,
and TileLang is disabled on SM75, so the fork ships `tools/flashqla_sm75_patches/gdn_forward.cu`:
a hand-written SIMT forward recurrence (`gdn_forward`, `gdn_forward_varlen`) with template
`<scalar_t, gate_t, state_t, D, COLS, WIDTH>`, per-token sequential state in registers and
subgroup (`__shfl_*`) reductions, JIT-built by `sm_legacy.py` into an extension that must export
both symbols (a stale binary is rejected with an explicit "re-run build.sh" error). The GDN layer
selects it through `gdn_prefill_backend=flashqla_legacy` and otherwise falls back to stock vLLM.

Kernel organization, from the file itself:

- grid is `(v_heads, batch, column-groups)`: one block owns one value head of one sequence and
  the `D` columns are split across blocks along `blockIdx.z`;
- each warp splits into subgroups of `WIDTH` lanes (`subgroups_per_warp = 32 / WIDTH`), and every
  lane keeps `rows_per_lane = ceil(D / WIDTH)` state rows for `COLS` columns in a
  `float state_shard[COLS][rows_per_lane]` register array -- the whole recurrence stays in
  registers, with `__shfl_down_sync`/`__shfl_sync` reductions inside the subgroup;
- dispatch: `D == 128` uses `COLS=4, WIDTH=16`; other head dimensions use `COLS=1, WIDTH=32`;
  each block then covers 8 column groups;
- the varlen entry is a packed-batch variant driven by `cu_seqlens` with an `input_batch` flag
  for the layout vLLM's GDN layer actually passes; the loader validates the extension against
  `_REQUIRED_EXTENSION_SYMBOLS = ("gdn_forward", "gdn_forward_varlen")`;
- the GDN layer logs which backend won (`Using FlashQLA legacy SM70/SM75 GDN prefill kernel
  (requested=flashqla_legacy, head_k_dim=128)` in the local log) and keeps FlashInfer JIT,
  CuteDSL, and Triton/FLA as alternatives.

`benchmarks/replayssm/e2e_decode_speedup.py` in that tree is unrelated to this kernel: it
benchmarks ReplaySSM (recent-input caching for Mamba2 decode) against the standard SSM kernel on
Blackwell/Nemotron models, running each mode in a separate subprocess for a clean CUDA context.

### 7.5 Communication and graphs

- Custom all-reduce comes from `csrc/quickreduce` (an AMD-quickreduce derivative; the headers are
  HIP-flavored but compile for CUDA here). The two-shot kernel processes one 32 KiB tile per block
  (256 threads x 8 atoms x 16 B), dispatches on world size (2/4/8 in the macro table, 2-16 per the
  changelog), and ships five codecs: `CodecFP` plus quantized `CodecQ3/Q4/Q6/Q8`; INT3 is
  restricted to `world_size == 2` because on TP4/TP8 the pack/unpack overhead outweighs the
  traffic saved. Graph-safety detail: each block reads its flag color from `d_flag_counters`,
  advances it, and writes it back on-device, so colors keep changing across CUDA-graph replays
  instead of being frozen -- the same class of problem this checkout tracks as graph address and
  counter stability. The launcher can disable it per route (`--disable-custom-all-reduce`).
- CUDA graphs: `FULL_AND_PIECEWISE` is the fast default. The SM75-specific piece is the
  speculative FA2 prefill wrapper: `_sm75_spec_prefill_graph_query_len` enables it only when the
  device is (7,5), the dtype is FP16, decode-context parallelism is 1, non-causal attention and
  batch invariance are off, the speculative width is 1..7, decode graphs are FULL, the capture
  sizes are exactly `{q * i, i = 1..max_num_seqs}` with `max_cudagraph_capture_size = q *
  max_num_seqs` (`q = num_speculative_tokens + 1`), and every attention layer has head size
  128/256 with FP16/FP8/quantized (non-NVFP4) KV storage. When it applies, one wrapper per
  `(batch, query, causal)` is created with pinned planning buffers (`qo_indptr` and
  `paged_kv_indptr` of `batch + 1`, `paged_kv_indices` of `batch * ceil(max_model_len /
  page_size)`, `paged_kv_last_page_len` of `batch`), which is what lets the whole speculative
  prefill live inside a FULL graph. The local log shows the resulting wrapper for MTP-4:
  `batch=1, query=5, heads=12/2, head_dim=256, page_size=16, causal=True`.
- Other SM75 gates seen in this tree: `fa_utils.py` keeps FlashAttention symbols but falls back
  because "SM75 has no supported FA2/FA3/FA4 extension"; `rotary_embedding/common.py` and
  `fused_moe` carry similar Turing fallbacks; the NVFP4 KV path is excluded from the wrapper.

### 7.6 Prefix cache and Mamba state

Prefix caching (default on) is a first-class concern for them because the hybrid GDN models carry
recurrent state. `MambaCacheMode` has three values (`vllm/config/cache.py`):

- `none` -- set when prefix caching is disabled;
- `all` -- cache the Mamba state at every position `i * block_size`;
- `align` -- cache only the state of the last token of each scheduler step when that token sits
  at `i * block_size`; default when prefix caching is enabled, and what the Qwen routes use.

The rest of the mechanism: fine-grained Mamba prefix hits are restricted to backends that can
restore their recurrent state (GDN/FlashQLA uses **block-aligned** hits, FlashKDA keeps finer
reuse); `enable_mamba_fine_grained_prefix_cache` (off by default) additionally registers an
"align" checkpoint at the shared-prefix junction where an MTP/EAGLE sibling resumes; hashed align
boundaries survive decode while obsolete unhashed blocks are released; DFlash2 keeps its target
replay state across prefix reuse and alternating prefill/decode turns; target Mamba speculative
state pages are restored for DFlash2/DSpark while draft KV stays in its own pool; and
`mamba_block_size` is documented as needing to be a multiple of 8 to line up with the
`causal_conv1d` kernel.

`v0.2.2-post3` adds opt-in SSD prefix-KV persistence keyed by a checkpoint fingerprint
(`tools/checkpoint_fingerprint.py` hashes file identity -- device/inode/size/mtime/ctime -- for
the target and optional draft), validated across a host reboot and service restarts; Mooncake
Store is also a supported prefix-hit backend.

Three of their changelog entries form a useful checklist for anyone caching recurrent state:
FP8-KV hybrid prefix hits when the Mamba block size equals the hash block size (including their
1648-token SM75 route), keeping the resident replay / hashed-boundary / target-speculative page
accounting consistent in align mode, and preserving request-row boundaries plus FULL-graph
padding across speculative replay.

## 8. Published measurements and their exact environment

From `profiles/2x2080Ti/README.md` (validated 2026-09-19, release `v0.2.1`). Columns are
`4K/128 prefill / decode` and `32K/512 prefill / decode`, both in tok/s, single request.

| Route | Context | KV | Spec | GPU KV tokens | 4K/128 | 32K/512 |
|---|---:|---|---|---:|---:|---:|
| `w8a16/mtp4-fp16kv-1x148K` | 148K | FP16 | MTP/4 | 149,964 | 1643.37 / 97.55 | 1425.34 / 95.10 |
| `w8a16/nomtp-fp16kv-1x176K` | 176K | FP16 | none | 179,940 | 1694.94 / 33.17 | 1483.04 / 30.80 |
| `w8a16/nomtp-fp16kv-1x121K` | 121K | FP16 | none | 122,608 | 1697.78 / 32.81 | 1447.04 / 30.65 |
| `w8a16/mtp4-fp8kv-1x262K` | 262K | FP8 | MTP/4 | 295,455 | 1623.68 / 94.60 | 1393.94 / 92.47 |
| `w8a16/mtp4-fp8kv-1x186K` | 186K | FP8 | MTP/4 | 190,540 | 1634.73 / 96.11 | 1384.95 / 93.17 |
| `w8a16/yarn-fp8kv-1x338K` | 338K | FP8 | none + YaRN | 354,143 | 1239.30 / 23.71 | 1338.48 / 17.91 |
| `w4a16/dflash2-fp8kv-1x262K` | 262K | FP8 | DFlash2/7 | 285,081 | 1370.18 / **220.84** | 1264.66 / **209.35** |
| `w4a16/dflash-fp8kv-2x176K` | 2x176K | FP8 | DFlash/7 | 357,194 | 1412.94 / 223.08 | 1249.27 / 211.10 |
| `w4a16/mtp4-fp8kv-2x229K` | 2x229K | FP8 | MTP/4 | 468,978 | 1432.48 / 116.04 | 1247.12 / 109.72 |
| `w4a16/mtp4-tq4nc-3x262K` | 3x262K | TQ4NC | MTP/4 | 837,832 | 1446.95 / 127.86 | 1265.68 / 68.64 |
| `w4a16/yarn-fp8kv-1x524K` | 524K | FP8 | none + YaRN | 588,863 | 1466.29 / 41.36 | 1274.26 / 37.22 |

Their Qwen3.6-35B-A3B rows (`6690.07 / 113.63` and `5941.16 / 105.92` at 262K/FP16) and the 2xT10 /
4xT10 tables exist in the sibling profile guides.

Their measurement lane (`profiles/2x2080Ti/README.md`, notes): prefix cache disabled during the
test, **one text-only request at a time**, warm-up excluded, `4K/128` = median of three requests,
`32K/512` run to completion, raw request JSON kept in an external audit directory. "DFlash2 default
K=7" and the headline rows use synthetic text with high speculative-hit rates (their own caveat:
real-task throughput depends on acceptance).

Host for those numbers: dual-socket Intel Xeon E5-2673 v4 (80 logical CPUs), 60 GiB RAM + 8 GiB
swap, **physical GPUs 1 and 5** (2080 Ti 22,528 MiB) over **NV2**, driver 595.91.07, runtime
`vllm-def-cu130` = CUDA 13.0 + torch 2.13.0+cu130.

Three things follow for this checkout:

1. Those numbers are **not** this machine: different CPUs, different GPU pair, different driver and
   CUDA/torch, and a different KV/speculation mix. The local 2026-10-03 service start shows the
   stack does come up here, but no local throughput was recorded.
2. The FP16-KV row beating the FP8-KV row (95.10 vs 92.47) means the local FP8-lane gap cannot be
   attributed mainly to FP8 KV.
3. The 209-220 tok/s NVFP4/DFlash2 headlines are not comparable to this checkout's FP8 + MTP3
   workload; the FP8/MTP4 rows (`92-95 tok/s`) are the closer analogue, still on their host.

### 8.1 How those numbers are produced

The lane is scripted, which matters when comparing against our protocol:

- `tools/profile_request.py` builds **exact-length prompts**: a framing prefix and suffix are
  tokenized and the remaining budget is filled with repeated `" the"` tokens (a `pure_filler`
  variant drops the framing), then the prompt is clamped so the token count is exactly 4096 or
  32768. Image variants keep the same token count, ask an image question in the suffix, and send
  the image as base64. It streams the completion and reports TTFT plus
  `decode_tok_s = delivered_tokens_after_first / (elapsed - ttft)`; a monitor thread samples
  `nvidia-smi` throughout.
- The reference lane disables prefix caching only for the measurement, sends one text-only
  request at a time, excludes warm-up, takes the median of three 4K/128 requests, and runs
  32K/512 to completion. `4K/128` means exactly 4,096 input and 128 output tokens; `32K/512`
  means exactly 32,768 input and 512 output tokens.
- Raw request JSON and logs stay in an external audit directory, not in the repository.
- Their warm-up policy is the opposite of this checkout's: they exclude warm-up from a lane that
  runs one request at a time, while this host needs warm-up runs to leave the SM clock-ramp
  window before a shape's median is stable. Do not port conclusions across those protocols.

### 8.2 The other two platforms

`2xT10` (two Tesla T10 16 GiB over PCIe PIX, TP2; `qwen27b/w4a16` = RedHatAI INT4): MTP3 with
FP8 KV at 262K measures `1122.00 / 87.88` (4K/128) and `1070.10 / 83.57` (32K/512); the
TurboQuant variants trade 32K decode rate for capacity (TQK8V4 2x155K: `1000.18 / 59.55`;
TQ4NC 2x220K: `1100.51 / 49.76`).

`4xT10` (four T10, TP4): the fastest routes are DFlash2 -- W8A16 FP16-KV `1433.91 / 191.89` and
`1536.13 / 189.38`; W4A16 NVFP4 FP16-KV `1435.68 / 220.99` and `1505.42 / 216.84`; the
NVFP4 + TQK8V4 4x220K route reaches 923,137 GPU KV tokens at `GPU_UTIL=0.93`. Their notes
record that four simultaneous 32K requests passed, that four near-limit requests at once can
OOM on DFlash temporary buffers, and that under a two-request 32K load the W8A16 FP8-KV DFlash2
route measured 62.88 and 65.70 tok/s per request -- i.e. even their own tables show that
concurrency changes the decode picture, which is why the headline rows are single-request.

### 8.3 Local concurrency reproduction on this host (2026-10-07)

One local run exists, and it measures **concurrency**, not the published rows. Raw outputs are
under [`eval/results/vllm-fp8-concurrency/`](../../eval/results/vllm-fp8-concurrency/README.md).

Stack: `v0.2.2-post3`, the FP8 artifact (W8A16), TP2 on GPU `0,1` (NV2), fp8 KV, MTP4, mode `fast`.
Two overrides on top of the published profile: `MAX_MODEL_LEN=65536` and `MAX_NUM_SEQS=3` (the
launcher refuses a configuration whose pool cannot admit `MAX_NUM_SEQS x MAX_MODEL_LEN`; the pool
resolved to 219,934 tokens, so 3 x 65,536 = 196,608 was the largest admissible window), plus
thinking disabled per request. Every request carried a unique prefix, so no prompt ever hit the
prefix cache.

| Prompt tokens | C | Wall | TTFT | Per-request decode | Aggregate e2e | Aggregate steady decode |
|---:|---:|---:|---:|---:|---:|---:|
| 4,131 | 1 | 9.58 s | 2.77 s | 75.1 | 53.5 | 75.2 |
| 4,131 | 2 | 12.83 s | 5.53 / 4.90 s | 71.4 / 64.4 | 79.8 (1.49x) | 140.3 (1.87x) |
| 4,131 | 3 | 16.34 s | 5.71 / 5.05 / 8.63 s | 49.4 / 46.0 / 66.3 | 94.0 (1.76x) | 199.1 (2.65x) |
| 32,861 | 1 | 34.49 s | 26.89 s | 67.2 | 14.8 | 67.3 |
| 32,861 | 2 | 61.83 s | 53.56 / 52.99 s | 64.6 / 57.8 | 16.6 (1.12x) | 123.8 (1.84x) |
| 32,861 | 3 | 89.22 s | 53.10 / 53.10 / 80.20 s | 14.6 / 14.5 / 56.7 | 17.2 (1.16x) | 170.3 (2.53x) |

All rates are tok/s for 512 output tokens per request. "e2e" divides every output token by the
wall clock including the cold prefill; "steady decode" divides them by (last request end - last
TTFT), which excludes the interval in which one request is still prefilling.

What the run establishes:

- **Concurrency scales sub-linearly but consistently**: steady aggregate 1.87x / 2.65x at C=2 / C=3
  for the short prompt and 1.84x / 2.53x for the long one; the per-request equivalent falls from
  75.2 to 70.2 to 66.4 tok/s (short) and 67.3 to 61.9 to 56.8 (long).
- **Cold prefill is serialized**: with `MAX_BATCHED_TOKENS=2048`, TTFT is C x 26.9 s and the
  aggregate prefill rate holds at ~1,230 tok/s at every concurrency, so the long-prompt e2e figure
  is prefill-bound (14.8 -> 17.2), not decode-bound.
- **No KV pressure**: peak occupancy was 3 x 32,861 = 98,583 of 219,934 tokens (45%); no preemption
  and no error in the run log. 3 x 65,536 = 89% of the pool is this configuration's admittance
  limit, i.e. roughly 73K tokens per request at C=3.

What it does not establish: this is not a reproduction of the published rows. It differs in
version, `MAX_MODEL_LEN`, thinking mode and prompt text (repeated technical prose, so MTP
acceptance is not the published lane's), and it is a single measurement per point. The
matching-context single-request points (1,491 prefill / 75.1 decode at 4K; 1,264 / 67.2 at 32K) sit
8-20% below the published v0.2.1 `1623.68 / 94.60` and `1393.94 / 92.47`, and that gap is
unexplained.

## 9. Relationship to this checkout

| External feature | NInfer status |
|---|---|
| FP8 W8A16 via Marlin FP8 kernels (Python-side Turing gates lowered) | `fp8-block128` research lane with its own block-scale semantics and two persistent layouts; the prefill accumulation form was evaluated and adopted where the [performance page](../performance.md#external-sm75-reference-vllm-2080ti-definitive) records it |
| NVFP4 W4A16 through Marlin | `nvfp4` identity plus `groupwise-int`; Marlin-style persistent layout implemented here as an FP8 block-128 layout, not as NVFP4 W4A16 |
| MTP3/MTP4 (sequential draft through the checkpoint's MTP layer) | MTP3 + `--lm-head-draft`; MTP attention unified onto the fused projection leaf (history §63) |
| DFlash2 (cross-attention drafter on target hidden states, K=7) | DFlash block mode (`--spec dflash --draft-tokens N`, `k=7` supported) implemented natively for the 27B groupwise-int/GGUF packages and for 35B-A3B (text-only); no cross-attention-on-hidden-states drafter |
| DFlash2 candidate selector and grouped conv | This checkout already carries a `dflash_selector` Op (`include/ninfer/ops/dflash_selector.h`, `src/ops/launcher/dflash_selector.cu`) on the DFlash path; their selector adds learned predecessor/successor codebooks and per-layer predicted FIR convolutions around attention and the MLP |
| BF16-trained draft executed at FP16 with an explicit numeric transport | No analogue: our drafts run in the package's own dtype. If a BF16 draft ever has to run on FP16 Turing, `dflash2_sm75.py` is the reference for making the cross-dtype contract explicit (gains, RNE emulation, row scaling, TP scale domain) |
| TurboQuant KV (rotation + Lloyd-Max) | Not implemented. KV choices here are `fp16` and `int8` (`--kv-dtype`) |
| Mamba state modes (`all`/`align`) and block-aligned recurrent checkpoints | No analogue: GDN state here is request-local, and prefix reuse covers text only |
| FP8 KV | Available on other lanes; the active FP8 lane is defined with FP16 KV, and capacity/prompt occupancy must not be conflated |
| Chunked prefill + `max-num-seqs`, continuous batching, `FULL_AND_PIECEWISE` graphs | This Engine uses bounded FIFO ingress, one compact decode batch per round, startup-fixed 1-8 requests, no preemption, CUDA Graph capture per phase |
| Custom all-reduce (quickreduce), 2-16 GPUs | TP2 over NVLink P2P; no in-tree custom all-reduce |
| Prefix cache with Mamba align mode, SSD persistence | Prefix reuse exists for text; no SSD KV persistence |
| Launcher profiles + modes + `/health` + smoke test | CLI/`--help` and serving docs; no profile-mode concept |

## 10. Reuse rules

- Quote an external number only with its route (KV dtype, spec method/tokens, context, concurrency)
  and its host; never drop it into a local table without the "external" label.
- Prefer their `profiles/<hardware>/README.md` over the top-level README when citing numbers; the
  profile tables carry the environment note.
- Their SM75 findings worth re-checking before we touch related code: TileLang incompatibility,
  deadlock-prone FLA Triton RMSNorm under 4 concurrent SM75 ranks, CUTLASS FP8 dispatch needing a
  Turing fallback, FlashInfer FA2 graph-buffer stability, and the GDN "one scale domain" TP
  reduction (`dflash2_sm75.py`).
- Their engineering habits that transfer: guard every Turing path and fail closed (the GDN extension
  staleness check, the TQ adaptive-boundary fail-closed), keep build-time patches as explicit files,
  and require capacity **and** throughput evidence per route before promoting it.
- Do not read their PPL deltas (`+1.17%`, `+2.71%`, ...) as measurements of our routes: they belong
  to their quantizer and their evaluation. Use them only to rank their presets against each other.
- When comparing an external row with a local one, fix the five things that change between adjacent
  external rows before comparing anything else: KV dtype, speculative method/width, `MAX_NUM_SEQS`,
  `MAX_MODEL_LEN`, and `GPU_UTIL`. Several of their rows differ in capacity per sequence, not in
  kernel, and their DFlash2 rows additionally rest on synthetic high-acceptance prompts.
