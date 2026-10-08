# Tests

The retained tests protect current `.ninfer`, numerical operator, target, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

- `artifact/` — Python container, registered layout, quantization, and resource behavior;
- `ops/` — one identifiable qualification suite per semantic Op or closely related overload group,
  using independent numerical/state-transition oracles at real supported shapes;
- `ops/linear/` — weight/activation-profile-specific public Linear conformance tests plus their
  one shared input generator, FP64 GEMM oracle, tolerance registry, and output/effects mechanics;
- `ops/linear_add/`, `ops/linear_pair/`, `ops/linear_swiglu/` — fused-Op suites split by registered
  weight/activation profile, each evaluating its complete formula rather than composing production
  Ops;
- `ops/test_allreduce.cpp` — the two-device `allreduce_sum`/`allgather_rows` collectives; it needs
  two CUDA devices visible to one process and reports the shared skip code when fewer are present;
- `targets/qwen3_6/` — shared tokenizer/template, multimodal preprocessing, MRoPE, prepared-prompt,
  stop/output decoding, hybrid topology, decoder/GDN and round-state layouts/views, shifted-MTP
  alignment, Vision control, and family runtime mechanisms;
- `targets/qwen3_6_27b/` — registered inventory, converter recipe, source verifier, artifact
  bindings, reference diagnostics, family Program/multimodal/MTP behavior, and the opt-in real-Engine
  prefix test;
- `targets/qwen3_8_27b/` — Qwen3.8-specific conversion contracts, including official block-FP8;
- `targets/qwen3_6_35b_a3b/` — registered inventory/converter contracts, artifact-native diagnostic
  reference, MoE oracle, typed binding, selected-expert row access, 256K INT8 memory calculation,
  and the opt-in real public-Engine route;
- `test_ninfer_artifact_reader.cpp` — C++ framing, directory, encoded-size, payload-span, and
  geometry behavior against a self-contained C++ fixture;
- `test_request_memory.cpp` — startup-frozen request-transient capacity, stable address,
  activation alignment, rejection, and peak semantics;
- `test_openai_schema.cpp`, `test_responses_schema.cpp`, `test_response_store.cpp`,
  `test_anthropic_schema.cpp`, and `test_tool_call_parser.cpp` — current protocol translation,
  Responses Item/state/SSE behavior, and incremental tool-call behavior;
- `test_request_log.cpp` and `test_http_error_handler.cpp` — generation lifecycle records,
  preparation rejections, protocol-shaped payload-limit errors, and application-error preservation;
- `test_ninfer_bench_support.cpp` — product benchmark CLI, timing boundary, and schema-v9 reports;
- `test_bench_matrix.py` — schema-v9 report consumption by the Python matrix summarizer;
- `test_serve_corpus.py` — serving request-log schema compatibility at the measurement consumer;
- device/tensor/arena tests — reusable lower-component behavior; KV tests cover the core physical
  container, family runtime tests cover dimension-driven GDN storage/view mechanics, and Op tests
  cover mathematical state transitions at their own boundary.

Tests are grouped by observable risk, not by mirroring every source file or class.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/W8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_TESTING=ON
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

Run a focused target for a localized change:

```bash
cmake --build build --parallel 4 --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|gqa_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel 4 --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_w8_a16_test
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|w8)_a16_test$' --output-on-failure
```

The Q4/Q5/Q6/W8 profile suites use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

### Block-FP8 checks

The BF16 vocabulary head is covered by `ninfer_linear_bf16_a16_test` at both full/TP2 shapes,
including T1 all-row FP64 comparison and T2-T5 route boundaries. Its host oracle uses at most
eight worker threads. To check FP8 MTP state without loading the 30.6 GB artifact on one card:

```bash
NINFER_QWEN3_8_27B_WEIGHTS=/data/models/qwen3.8-27b/qwen3_8_27b_fp8_block128.ninfer \
NINFER_MTP_TP2_FP16_ONLY=1 ./build/tests/ninfer_qwen3_8_27b_mtp_tp2_real_test
```

This existing suite's explicit TP2/FP16 slice checks repeated generation, graph/eager tokens,
cross-rank acceptance/commit egress and fixed-prefix teacher forcing against the ordinary
target, using the same existing near-tie criterion. The default TP1/TP2/INT8 suite is unchanged.

For SM75 FP16 prefill attention work, run
`NINFER_GQA_FP16_ONLY=1 ./build/tests/ninfer_gqa_attention_test` to select the existing FP16
mathematical, cache and batch checks, including the TP2 T129 fragmented-page tail. The default
suite still runs both KV formats; this selection does not qualify or suppress INT8 failures.

| Existing entry | What it protects |
|---|---|
| [test_fp8_block_numeric.py](artifact/test_fp8_block_numeric.py) | exact code/scale round trip, partial 128x128 tiles, multiplier decode and invalid words |
| [test_fp8_block_converter.py](targets/qwen3_8_27b/test_fp8_block_converter.py) | registered source recipe and conversion transforms |
| [test_fp8_block.cpp](ops/linear/test_fp8_block.cpp) / `ninfer_linear_fp8_block_test` | independent FP64 Linear, LinearAdd, SwiGLU, attention/GDN projection and LinearPair formulas; optional real Text/MTP weights and TP2 slice bytes |
| [materialization real test](targets/qwen3_6_27b/test_fp8_block_materialization_real.cpp) | actual target plan upload/readback; `ninfer_qwen3_8_27b_fp8_block_materialization_real_test` |
| [logits real probe](targets/qwen3_6_27b/test_fp8_block_logits_real.cpp) | exact artifact embedding-row capture gate, diagnostic last-prefill logits/layers and sampling boundary; `ninfer_qwen3_8_27b_fp8_block_logits_real_test` |

```bash
python3 -m pytest -q tests/artifact/test_fp8_block_numeric.py \
  tests/targets/qwen3_8_27b/test_fp8_block_converter.py
cmake --build build --parallel 4 --target ninfer_linear_fp8_block_test
NINFER_FP8_BLOCK_ARTIFACT=/data/models/qwen3.8-27b/qwen3_8_27b_fp8_block128.ninfer \
NINFER_FP8_BLOCK_ROUTE=hmma ./build/tests/ninfer_linear_fp8_block_test
```

Run the last command separately with `NINFER_FP8_BLOCK_ROUTE=scalar` for the other implementation.
Run it with the route variable unset for the production auto thresholds.
Without the artifact variable the synthetic cases still run, but real-weight cases do not.
Synthetic cases cover T4/5/8/9 and GDN T12/13. T192 covers Linear, SwiGLU and attention/GDN
scatter, including sampled columns127/128. Real Text/MTP weights cover T1/4/5/9/192;
real GDN and `5120x17408` down Linear cover T12/13. All use synthetic BF16 activations,
not real activation replay. The three routes use the same independent FP64 formulas and
unchanged absolute/relative criteria; passing these does not qualify complete model behavior.
These are SM75 W8A16 checks, not the architecture-stubbed FP8 A8 tests. A repository-wide CTest
failure from an unsupported A8/A4 leaf is not evidence that block-FP8 failed its oracle.

The materialization test requires `NINFER_FP8_BLOCK_ARTIFACT` and two devices. The logits probe
additionally requires `NINFER_FP8_BLOCK_PROMPT_IDS`; set `NINFER_FP8_BLOCK_ROUTE_COMPARE=1` to
compare the same artifact (defaults scalar/HMMA), otherwise it also requires
`NINFER_GROUPWISE_INT_ARTIFACT`. `NINFER_FP8_BLOCK_DEVICES=0,1` and
`NINFER_FP8_BLOCK_PREFILL_CHUNK=4096` fix the route-compare research configuration; `_ROUTE_A/B`
and `_PREFILL_CHUNK_A/B` variants provide explicit controls. Missing prerequisites return skip77.
Every FP8 capture checks its final token embedding against the artifact's contiguous BF16 row,
word for word. A mismatch fails the probe even when both routes repeat the same stale data;
reported post-mixer/logit distances remain diagnostic, not a model-quality gate.

The probe's route-compare branch reports errors without an error-threshold assertion. Exit0 is
not a numerical pass. Its capture is the prefill-finalization boundary, not every decode position;
later comparisons require an explicit common token prefix. Embedding capture producer ordering
is still under investigation, and groupwise weights are not a same-source oracle. See
[todo.md](../todo.md) for the minimal regression and model-quality checks still needed.

Use the selected Python3.11 environment with existing torch/pytest. A prior local attempt lacked
pytest; the commands above are entry points, not a claim that this documentation update ran them.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/targets/qwen3_6_27b tests/targets/qwen3_6_35b_a3b \
  tests/test_bench_matrix.py tests/test_serve_corpus.py
```

The Python binding tests use `NINFER_QWEN3_6_27B_ARTIFACT` when set, otherwise they look for
`out/qwen3_6_27b.ninfer`. They report a pytest skip when neither path provides the real
artifact. The 35B-A3B reference binding test follows the same rule with
`NINFER_QWEN3_6_35B_A3B_ARTIFACT` and `out/qwen3_6_35b_a3b.ninfer`. The remaining Python
target tests still run without either artifact.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_prefix_real_test --output-on-failure
```

Run the peer 35B-A3B route independently:

```bash
NINFER_QWEN3_6_35B_A3B_WEIGHTS=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_35b_a3b_real_test --output-on-failure
```

Without the corresponding variable CTest marks each C++ integration test as skipped. Neither test
uses another numerical/execution path's generated tokens as a golden.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares restored and cold greedy output, compares stripped and preserved closed-turn
prompt lengths, and verifies turn/response rewrite-checkpoint reuse paths plus Responses
inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact registered artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- family Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.
