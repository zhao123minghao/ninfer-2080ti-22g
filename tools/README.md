# NInfer maintainer tools

`tools/` contains the project-owner workflows for artifact conversion and inspection, independent
Python references, numerical parity diagnostics, benchmark orchestration, and serving smoke checks.
These tools are not part of the public download-and-run path; normal users should start with the
[project README](../README.md).

Run commands from the repository root with a Python 3.11 environment containing the dependencies
for the selected tool.

## Task index

| Task | Location |
|---|---|
| Build the 27B artifact | [`convert/qwen3_6_27b/`](convert/qwen3_6_27b/) |
| Build the Qwen3.8-27B artifact | [`convert/qwen3_8_27b/`](convert/qwen3_8_27b/) |
| Preserve official Qwen3.8 block-FP8 weights | [`convert/qwen3_8_27b/convert_fp8_block.py`](convert/qwen3_8_27b/convert_fp8_block.py) |
| Dump same-source fixed-token next-token scores | [`parity/qwen3_8_27b/dump_source_logits.py`](parity/qwen3_8_27b/dump_source_logits.py) |
| Dump a same-source greedy trajectory with per-position scores | [`parity/qwen3_8_27b/dump_source_trajectory.py`](parity/qwen3_8_27b/dump_source_trajectory.py) |
| Export Engine top-k over fixed token-ID prefixes (the other half of that oracle) | [`parity/qwen3_8_27b/engine_logits_probe.cpp`](parity/qwen3_8_27b/engine_logits_probe.cpp) |
| Generate quality-gate prompt token-ID files (short text / code / reasoning) | [`parity/qwen3_8_27b/make_quality_prompts.py`](parity/qwen3_8_27b/make_quality_prompts.py) |
| Compare a source trajectory report against an Engine logits report | [`parity/qwen3_8_27b/compare_quality.py`](parity/qwen3_8_27b/compare_quality.py) |
| Compare same-source TP2 FP8 Linear with Marlin | [`parity/qwen3_8_27b/compare_marlin_linear.py`](parity/qwen3_8_27b/compare_marlin_linear.py) |
| Build the 35B-A3B artifact | [`convert/qwen3_6_35b_a3b/`](convert/qwen3_6_35b_a3b/) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Run the 27B Python reference | [`reference/qwen3_6_27b/`](reference/qwen3_6_27b/README.md) |
| Run the 35B-A3B Python reference | [`reference/qwen3_6_35b_a3b/`](reference/qwen3_6_35b_a3b/README.md) |
| Compare 27B artifact/source Vision activations | [`parity/qwen3_6_27b/`](parity/qwen3_6_27b/README.md) |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |

## Artifact workflow

The groupwise-int converters consume official local BF16 checkpoints. The NVFP4, preserved-GGUF
and official block-FP8 converters instead follow their own fixed source contracts. Each writes one
complete `.ninfer` artifact. The paths below are placeholders for local checkpoints:

```bash
python3 -m tools.convert.qwen3_6_27b.convert \
  --model /path/to/Qwen3.6-27B \
  --out out/qwen3_6_27b.ninfer

python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --out out/qwen3_8_27b.ninfer

python3 -m tools.convert.qwen3_6_35b_a3b.convert \
  --model /path/to/Qwen3.6-35B-A3B-base \
  --dflash-model /path/to/Qwen3.6-35B-A3B-DFlash \
  --out out/qwen3_6_35b_a3b.ninfer
```

Inspect either result:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

The exact source revisions, inventories, formats, and conversion recipes are recorded in
[`docs/maintainer/`](../docs/maintainer/). Published users download the completed artifacts from
Hugging Face instead of running these workflows.

### Official block-FP8 conversion

```bash
python3 -m tools.convert.qwen3_8_27b.convert_fp8_block \
  --model /data/models/qwen3.8-27b/Qwen3.8-27B-FP8 \
  --out /data/models/qwen3.8-27b/qwen3_8_27b_fp8_block128.ninfer \
  --device cuda
```

`--marlin-layout` writes the same logical FP8 code/BF16 multiplier words in
`marlin-fp8-block128-v1`, verifying every encoded FP8 object by decode-before-write.
It is a converter/research artifact mode until the native Marlin execution leaf is selected; do
not replace the registered block128 artifact or enable it in a production Engine prematurely.

This preserves 407 source FP8 matrices as 260 fused block128 objects, including exact BF16
`weight_scale_inv` multiplier words. It does not invert scales or requantize the FP8 codes.
The resulting identity is `qwen3.8-27b/fp8-block128`, not the row-scaled FP8 inside `nvfp4`.
See the [artifact contract](../docs/maintainer/qwen3.8-27b-artifact.md#15-official-block-fp8-artifact).
The local output is approximately 30.6 GB; do not regenerate it merely to run diagnostics.

The source-logits helper requires its compatible vLLM environment and fixed token IDs; use its
`--help` for arguments. Top-k logprob differences support margin comparisons, not recovery of a
complete exact raw-logit vector. The real Engine capture probe also needs stream-ordering checks
before layer differences are trusted. [todo.md](../todo.md) records these unresolved boundaries;
neither tool is an end-to-end quality gate by itself.

## Python references and parity

```bash
python3 -m tools.reference.qwen3_6_27b \
  --weights out/qwen3_6_27b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 128

python3 -m tools.reference.qwen3_6_35b_a3b \
  --weights out/qwen3_6_35b_a3b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 128
```

The Python implementations are independent diagnostic references, not alternate public inference
products or generated-token goldens for the C++ engine. See the parity README for the direct 27B
artifact/source Vision comparison command.

## Benchmark orchestration

For an operator-only Marlin comparison using the already-installed local vLLM environment:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
/home/zhaomh/miniconda3/envs/vllm/bin/python -m tools.parity.qwen3_8_27b.compare_marlin_linear \
  --model /data/models/qwen3.8-27b/Qwen3.8-27B-FP8 \
  --artifact /data/models/qwen3.8-27b/qwen3_8_27b_fp8_block128.ninfer \
  --bench ./build/bench/ninfer_linear_bench --out .scratch/marlin-compare \
  --matrix gate-up --rank 0 --tokens 1,4,5,4096 --warmup 5 --repeat 31
```

`--matrix down` selects the real TP2 K-half MLP down shard. The tool audits source/artifact
code and BF16-scale exactness and checks sampled outputs with an independent FP64 formula.
Default Linear uses a synthetic BF16 pattern exactly castable to Marlin FP16;
`--activation-file PATH` replays actual contiguous BF16 `[T,K]` words and reports cast error.
Marlin Linear produces FP16 and NInfer BF16; each is checked directly against original inputs,
not by pairwise equality. `--op swiglu` checks the complete FP64 SiLU/product formula;
`--op add --matrix down --residual-file PATH` checks the complete residual sum. Composed
Marlin routes include BF16-to-FP16 and FP32 epilogue/BF16 output in timing; they do not copy
private intermediate precision into the oracle. Report interpreter/torch/CUDA versions;
the local Marlin environment used for the recorded run is Python 3.12, not the maintainer 3.11
baseline. `--profile` with one token width brackets one post-warmup Marlin call for software
timeline/resource inspection; profiled times are not the normal timing results.

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```
