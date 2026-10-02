#!/usr/bin/env bash
# Dual-GPU (tensor-parallel) launcher for qwen3.8-27b on two RTX 2080 Ti (sm_75).
#
# Defaults target the 2x2080Ti / NVLink setup with the 262144-token context the
# build is qualified for:
#
#   MODEL        artifact path          (env NINFER_MODEL)
#   DEVICES      0,1                    (env NINFER_DEVICES)
#   TP           2                      (env NINFER_TP)
#   MAX_CONTEXT  262144                 (env NINFER_MAX_CONTEXT)
#   KV_DTYPE     fp16                   (env NINFER_KV_DTYPE; int8 for more KV capacity)
#
# Usage:
#   scripts/run.sh "your prompt"                 # greedy-ish defaults, your prompt
#   scripts/run.sh --messages chat.json --temperature 0.7
#   echo "your prompt" | scripts/run.sh          # prompt from stdin
#
# Any extra flags are forwarded after the injected defaults, so a later scalar
# flag (e.g. --max-context 4096) overrides the injected one.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${NINFER_MODEL:-/data/models/qwen3.8-27b/qwen3_8_27b_v2.ninfer}"
BIN="${NINFER_BIN:-./build/apps/ninfer}"
DEVICES="${NINFER_DEVICES:-0,1}"
TP="${NINFER_TP:-2}"
MAX_CONTEXT="${NINFER_MAX_CONTEXT:-262144}"
KV_DTYPE="${NINFER_KV_DTYPE:-fp16}"

if [[ ! -x "$BIN" ]]; then
    echo "run.sh: binary not found: $BIN" >&2
    echo "build it first: cmake --build build -j" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "run.sh: artifact not found: $MODEL" >&2
    echo "set NINFER_MODEL=/path/to/model.ninfer" >&2
    exit 1
fi

extra=()
# A first non-flag argument is treated as the prompt; everything else is forwarded
# verbatim so `--messages`, `--temperature`, `--greedy`, `--no-cuda-graph`, ... all
# still work.
if [[ $# -gt 0 && "$1" != -* ]]; then
    extra+=(--prompt "$1")
    shift
fi
# If stdin is piped (not a terminal) and no prompt/messages flag was given, read
# the prompt from it.
if [[ ! -t 0 ]]; then
    stdin="$(cat)"
    if [[ -n "${stdin//[[:space:]]/}" ]]; then
        extra+=(--prompt "$stdin")
    fi
fi

exec "$BIN" "$MODEL" \
    --devices "$DEVICES" --tp "$TP" \
    --max-context "$MAX_CONTEXT" --kv-dtype "$KV_DTYPE" \
    "${extra[@]}" "$@"
