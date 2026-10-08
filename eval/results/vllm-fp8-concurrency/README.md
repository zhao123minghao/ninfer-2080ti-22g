# vLLM 2080 Ti Definitive: local FP8 concurrency run (2026-10-07)

Raw outputs behind the concurrency measurement recorded in
[`docs/maintainer/vllm-2080ti-definitive-reference.md` §8.3](../../../docs/maintainer/vllm-2080ti-definitive-reference.md#83-local-concurrency-reproduction-on-this-host-2026-10-07).
This is the first throughput measurement of that stack taken on this host; it is a concurrency
run, not a reproduction of the fork's published rows.

## Conditions

| Setting | Value |
|---|---|
| Stack | `vLLM 2080 Ti Definitive Edition v0.2.2-post3`; `.venv` Python 3.12.13 / torch 2.13.0+cu130 / CUDA 13.0 |
| Weights | `Qwen/Qwen3.8-27B-FP8` (W8A16), `/data/models/qwen3.8-27b/Qwen3.8-27B-FP8` |
| Topology | TP2 on GPU `0,1` (NV2), on this host's 3 x RTX 2080 Ti 22 GB |
| Cache | fp8 KV; GPU pool 219,934 tokens (5.22 GiB per card at `GPU_UTIL=0.96`) |
| Speculation | MTP4 (`method=mtp, num_speculative_tokens=4`) |
| Overrides vs the published profile | `MAX_MODEL_LEN=65536`, `MAX_NUM_SEQS=3`; thinking disabled per request |
| Scheduling | chunked prefill, `MAX_BATCHED_TOKENS=2048`, mode `fast`, CUDA Graph `FULL_AND_PIECEWISE` with capture sizes `[5,10,15]` |
| Sampling | `temperature 0`, `ignore_eos`, 512 output tokens per request |
| Prefix reuse | defeated per request with a unique 8-byte prefix, matching the published lane's "prefix cache disabled during measurement" |
| Measurement | one run per point; 4K/512 single request repeated twice (75.1 / 76.6 tok/s) |

## Results

| Prompt tokens | C | Wall | TTFT | Per-request decode | Aggregate end-to-end | Aggregate steady decode |
|---:|---:|---:|---:|---:|---:|---:|
| 4,131 | 1 | 9.58 s | 2.77 s | 75.1 | 53.5 | 75.2 |
| 4,131 | 2 | 12.83 s | 5.53 / 4.90 s | 71.4 / 64.4 | 79.8 | 140.3 |
| 4,131 | 3 | 16.34 s | 5.71 / 5.05 / 8.63 s | 49.4 / 46.0 / 66.3 | 94.0 | 199.1 |
| 32,861 | 1 | 34.49 s | 26.89 s | 67.2 | 14.8 | 67.3 |
| 32,861 | 2 | 61.83 s | 53.56 / 52.99 s | 64.6 / 57.8 | 16.6 | 123.8 |
| 32,861 | 3 | 89.22 s | 53.10 / 53.10 / 80.20 s | 14.6 / 14.5 / 56.7 | 17.2 | 170.3 |

Rates are tok/s. `end-to-end` divides every output token by the wall clock including the cold
prefill; `steady decode` divides them by (last request end - last TTFT), which excludes the
interval in which one request is still prefilling. KV occupancy peaks at 3 x 32,861 = 98,583 of
219,934 tokens (45%); the run log recorded no preemption and no error.

## Files

| File | What it is |
|---|---|
| `out_4k_cold.txt`, `out_32k_cold.txt` | full client output for both prompt lengths |
| `bench.py` | the client: unique prefix per request, streaming TTFT, end-to-end and steady aggregate rates |
| `launch-command.txt` | the launcher invocation and the resolved `api_server` command line |
| `preflight-config.txt` | `--print-config` output confirming the profile plus both overrides |

The server was still holding GPU `0,1` when this directory was written. Stop it from the fork
checkout with `kill $(cat run-logs/vllm-Qwen3.8-27B-FP8.pid)`.
