# NInfer documentation

Start with the [project README](../README.md) for the Turing (`sm_75`) build on two 2080 Ti cards,
its TP2 groupwise-int acceptance result, the `--kv-dtype fp16|int8|fp8` cache choice and the
separate official block-FP8 research status. It also documents the published-artifact CLI and HTTP
examples and local conversions.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, dual-GPU (`--tp 2`) execution, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, tool calls, dual-GPU serving, and YaRN extended context |
| [Performance](performance.md) | Turing 85K-occupancy acceptance method and measurement status, the concurrent-decode, KV-cache dtype, FP8 and TP2 transport evidence; V100X2 tables and an archived one-line summary of the inherited RTX 5090 campaigns |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `gguf-q4-k-m` (local conversion) | generated locally as `qwen3_8_27b_q4_k_m.ninfer` | Text/MTP only; embedded Vision objects are validation-only |
| Qwen3.8-27B | `fp8-block128` (local research) | local official FP8 conversion | [artifact contract](maintainer/qwen3.8-27b-artifact.md#15-official-block-fp8-artifact); quality/performance acceptance open |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Maintainer tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

Runtime and Op references:

- [Small-scale concurrent inference architecture](maintainer/concurrent-inference-architecture.md)
- [Paged KV context storage, ownership, and capacity model](maintainer/paged-kv-cache.md)
- [Op admission, contracts, ownership, qualification, and performance rules](maintainer/op-development.md)
- [ReplaySSM GDN technical reference](maintainer/replayssm-gdn.md)
- [Dual-GPU (TP2) execution and YaRN 1M context](maintainer/tp2-yarn-1m.md)
- [Linear benchmark contract and registered suites](maintainer/linear-benchmark.md)

Artifact and model references:

- [NInfer artifact container](maintainer/artifact-container.md)
- [Persistent tensor numeric formats](maintainer/tensor-formats.md)
- [Persistent storage layouts](maintainer/storage-layouts.md)
- [Qwen3.6-27B model semantics](maintainer/qwen3.6-27b-model.md)
- [Qwen3.6-27B artifact contracts, including NVFP4](maintainer/qwen3.6-27b-artifact.md)
- [Qwen3.8-27B artifact contracts: groupwise-int, NVFP4, GGUF and official block-FP8](maintainer/qwen3.8-27b-artifact.md)
- [Qwen3.6-35B-A3B model semantics](maintainer/qwen3.6-35b-a3b-model.md)
- [Qwen3.6-35B-A3B artifact contracts](maintainer/qwen3.6-35b-a3b-artifact.md)

External reference projects:

- [vLLM 2080 Ti Definitive Edition](maintainer/vllm-2080ti-definitive-reference.md): the
  third-party SM75 serving stack this checkout compares against. Records its build and launch
  pipeline, Turing patch surface, MTP/DFlash2/TurboQuant algorithms, and the host behind each
  published number.

Pending implementation work:

- [Current FP8 investigations](../todo.md): capture ordering, current-route numerical coverage,
  same-source model checks and post-fix decode attribution. [History](../history.md) retains
  measurements with explicit corrections to superseded interpretations.
- [Softmax Attention organization and migration](maintainer/softmax-attention.md) describes the
  single target state for an unfinished source and public-contract cutover; it is not the current
  implementation map.
