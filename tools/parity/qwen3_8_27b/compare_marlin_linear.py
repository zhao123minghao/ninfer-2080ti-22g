"""Compare real TP2 MLP Linear inputs with Marlin and NInfer on one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
from safetensors import safe_open
import torch

from tools.artifact.container import Artifact


def bf16_values(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


def fp8_table() -> np.ndarray:
    values = []
    for code in range(256):
        sign = -1 if code & 128 else 1
        exponent = (code >> 3) & 15
        mantissa = code & 7
        if exponent == 15 and mantissa == 7:
            values.append(float("nan"))
        elif exponent == 0:
            values.append(sign * math.ldexp(mantissa, -9))
        else:
            values.append(sign * math.ldexp(1 + mantissa / 8, exponent - 7))
    return np.asarray(values, dtype=np.float64)


def source_shard(model: Path, rank: int, matrix: str) -> tuple[torch.Tensor, torch.Tensor]:
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    codes = []
    scales = []
    projections = ("gate_proj", "up_proj") if matrix == "gate-up" else ("down_proj",)
    for projection in projections:
        prefix = f"model.language_model.layers.0.mlp.{projection}"
        with safe_open(model / index[prefix + ".weight"], framework="pt", device="cpu") as shard:
            weight = shard.get_tensor(prefix + ".weight")
            scale = shard.get_tensor(prefix + ".weight_scale_inv")
        expected_shape = (17408, 5120) if matrix == "gate-up" else (5120, 17408)
        expected_scales = (136, 40) if matrix == "gate-up" else (40, 136)
        if weight.shape != expected_shape or scale.shape != expected_scales:
            raise ValueError("unexpected official MLP shape")
        if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.bfloat16:
            raise ValueError("unexpected official gate/up format")
        if matrix == "gate-up":
            codes.append(weight.view(torch.uint8)[rank * 8704:(rank + 1) * 8704])
            scales.append(scale[rank * 68:(rank + 1) * 68])
        else:
            codes.append(weight.view(torch.uint8)[:, rank * 8704:(rank + 1) * 8704])
            scales.append(scale[:, rank * 68:(rank + 1) * 68])
    return torch.cat(codes).contiguous(), torch.cat(scales).contiguous()


def audit_artifact(path: Path, codes: torch.Tensor, scales: torch.Tensor, rank: int,
                   matrix: str) -> None:
    with Artifact(path) as artifact:
        obj = artifact.find("text/layers/0/mlp/" + ("gate_up" if matrix == "gate-up" else "down"))
        payload = bytes(artifact.payload(obj))
    parent_rows, parent_columns = obj.shape
    code_bytes = parent_rows * parent_columns
    parent_codes = np.frombuffer(payload, dtype=np.uint8, count=code_bytes).reshape(obj.shape)
    parent_scales = np.frombuffer(payload, dtype="<u2", offset=code_bytes).reshape(
        parent_rows // 128, parent_columns // 128)
    if matrix == "gate-up":
        begin = rank * 8704
        expected_codes = np.concatenate((parent_codes[begin:begin + 8704],
                                         parent_codes[17408 + begin:17408 + begin + 8704]))
        scale_begin = rank * 68
        expected_scales = np.concatenate((parent_scales[scale_begin:scale_begin + 68],
                                          parent_scales[136 + scale_begin:136 + scale_begin + 68]))
    else:
        expected_codes = parent_codes[:, rank * 8704:(rank + 1) * 8704]
        expected_scales = parent_scales[:, rank * 68:(rank + 1) * 68]
    if not np.array_equal(codes.numpy(), expected_codes):
        raise ValueError("source/artifact TP2 codes are not exact")
    if not np.array_equal(scales.view(torch.uint16).numpy(), expected_scales):
        raise ValueError("source/artifact TP2 scale words are not exact")


def oracle(codes: torch.Tensor, scales: torch.Tensor, activation: torch.Tensor,
           operation: str = "linear", residual: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
    output_rows = codes.shape[0] // 2 if operation == "swiglu" else codes.shape[0]
    candidates = [0, 1, 127, 128, 255, 4095, 4351, 4352,
                  8191, 8703, 8704, 8831, 8832, 16383, output_rows - 2, output_rows - 1]
    rows = np.asarray(sorted({row for row in candidates if row < output_rows}))
    weight_rows = np.concatenate((rows, rows + output_rows)) if operation == "swiglu" else rows
    weight = fp8_table()[codes.numpy()[weight_rows]]
    multiplier = bf16_values(scales.view(torch.uint16).numpy())[weight_rows // 128]
    weight *= np.repeat(multiplier, 128, axis=1)
    represented = bf16_values(activation.view(torch.uint16).numpy()).astype(np.float64)
    result = represented @ weight.T
    if operation == "swiglu":
        gate, up = np.split(result, 2, axis=1)
        result = gate / (1 + np.exp(-gate)) * up
    elif operation == "add":
        result += bf16_values(residual.view(torch.uint16).numpy())[:, rows]
    return rows, result


def check_output(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    error = np.abs(actual.astype(np.float64) - reference)
    limit = 0.005 + 0.004 * np.abs(reference)
    return {
        "checked_values": int(error.size),
        "max_abs": float(error.max()),
        "rel_l2": float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-12)),
        "worst_error_to_limit": float((error / limit).max()),
        "passed": bool(np.isfinite(actual).all() and np.all(error <= limit)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--matrix", choices=("gate-up", "down"), default="gate-up")
    parser.add_argument("--tokens", default="1,4,5,4096")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=31)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--activation-file", type=Path)
    parser.add_argument("--op", choices=("linear", "swiglu", "add"), default="linear")
    parser.add_argument("--residual-file", type=Path)
    args = parser.parse_args()
    tokens = [int(value) for value in args.tokens.split(",")]
    if min(tokens) <= 0 or args.repeat <= 0 or args.warmup < 0:
        raise ValueError("invalid token or measurement counts")
    if args.profile and len(tokens) != 1:
        raise ValueError("--profile requires one token width")
    if args.op == "swiglu" and args.matrix != "gate-up":
        raise ValueError("SwiGLU requires gate/up weights")
    if args.op == "add" and (args.matrix != "down" or args.residual_file is None):
        raise ValueError("LinearAdd requires down weights and represented residual input")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.cuda.set_device(0)

    import vllm
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin,
    )

    codes, scales = source_shard(args.model, args.rank, args.matrix)
    audit_artifact(args.artifact, codes, scales, args.rank, args.matrix)
    output_rows, input_columns = codes.shape
    print("source/artifact TP2 code/scale exact", flush=True)
    with (args.out / "weight.bin").open("wb") as output:
        output.write(codes.numpy().tobytes())
        output.write(scales.view(torch.uint16).numpy().tobytes())
    if args.activation_file is None:
        activation = ((torch.arange(max(tokens) * input_columns, dtype=torch.int32) % 257 - 128)
                      .float().reshape(max(tokens), input_columns) / 256).to(torch.bfloat16)
    else:
        words = np.fromfile(args.activation_file, dtype="<u2", count=max(tokens) * input_columns)
        if words.size != max(tokens) * input_columns:
            raise ValueError("activation file is shorter than the requested represented input")
        activation = torch.from_numpy(words.reshape(max(tokens), input_columns)).view(torch.bfloat16)
    activation.view(torch.uint16).numpy().tofile(args.out / "activation.bin")
    result_rows = output_rows // 2 if args.op == "swiglu" else output_rows
    residual = None
    if args.op == "add":
        words = np.fromfile(args.residual_file, dtype="<u2", count=max(tokens) * result_rows)
        if words.size != max(tokens) * result_rows:
            raise ValueError("residual file is shorter than the requested represented input")
        residual = torch.from_numpy(words.reshape(max(tokens), result_rows)).view(torch.bfloat16)
        residual.view(torch.uint16).numpy().tofile(args.out / "residual.bin")
    represented = bf16_values(activation.view(torch.uint16).numpy())
    cast_values = activation.to(torch.float16).to(torch.float32).numpy()
    cast_error = np.abs(cast_values - represented)
    input_profile = {
        "source": str(args.activation_file) if args.activation_file else "synthetic pattern",
        "max_abs": float(np.abs(represented).max()),
        "fp16_cast_max_abs": float(cast_error.max()),
        "fp16_cast_exact": bool(np.array_equal(cast_values, represented)),
    }
    print("input profile " + json.dumps(input_profile), flush=True)

    layer = torch.nn.Module()
    layer.input_size_per_partition = input_columns
    layer.output_size_per_partition = output_rows
    layer.weight_block_size = (128, 128)
    layer.orig_dtype = torch.float16
    layer.weight = torch.nn.Parameter(codes.cuda().view(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scales.cuda(), requires_grad=False)
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    inputs = activation.cuda().to(torch.float16)
    represented_inputs = activation.cuda() if args.op != "linear" else None
    residual_device = residual.cuda().float() if residual is not None else None
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    report = {
        "source": str(args.model), "artifact": str(args.artifact), "rank": args.rank,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0), "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "shape": {"n": output_rows, "k": input_columns}, "matrix": args.matrix,
        "op": args.op, "output_rows": result_rows,
        "cache": "cold-256MiB", "warmup": args.warmup, "repeat": args.repeat,
        "input": input_profile,
        "source_artifact_exact": True, "criterion": "abs <= 0.005 + 0.004 * abs(FP64)",
        "ninfer_input_output": "BF16/BF16",
        "marlin_input_output": "FP16/FP16" if args.op == "linear" else "BF16/cast-FP16/BF16",
        "activation_cast_in_timing": args.op != "linear",
        "marlin_fp32_reduce": True,
        "points": [],
    }
    for width in tokens:
        rows, reference = oracle(codes, scales, activation[:width], args.op,
                     residual[:width] if residual is not None else None)
        point = {"tokens": width, "ninfer_us": [], "marlin_us": []}

        def run_ninfer(iteration: int) -> None:
            csv_path = args.out / f"ninfer_t{width}_{iteration}.csv"
            output_path = args.out / f"ninfer_t{width}.bf16"
            command = [str(args.bench.resolve()), "--qtype", "FP8BLOCK", "--policy", "a16",
                       "--n", str(output_rows), "--k", str(input_columns), "--t", str(width),
                       "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                       "--fp8-data", str(args.out.resolve()), "--output-bf16", str(output_path.resolve()),
                       "--epilogue", args.op,
                       "--csv-out", str(csv_path.resolve())]
            environment = dict(os.environ, NINFER_FP8_BLOCK_ROUTE="auto")
            subprocess.run(command, env=environment, check=True, stdout=subprocess.DEVNULL)
            with csv_path.open() as source:
                timing = next(csv.DictReader(source))
            point["ninfer_us"].append(float(timing["median_us"]))
            output_words = np.fromfile(output_path, dtype="<u2").reshape(width, result_rows)
            point["ninfer_oracle"] = check_output(bf16_values(output_words[:, rows]), reference)

        def run_marlin() -> None:
            durations = []
            output = None
            for iteration in range(args.warmup + args.repeat):
                flush.fill_(iteration & 255)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                if args.profile and iteration == args.warmup:
                    torch.cuda.cudart().cudaProfilerStart()
                start.record()
                current_inputs = (represented_inputs[:width].to(torch.float16)
                                  if represented_inputs is not None else inputs[:width])
                output = apply_fp8_marlin_linear(
                    current_inputs, layer.weight, layer.weight_scale_inv, layer.workspace,
                    output_rows, input_columns, bias=None, use_fp32_reduce=True,
                )
                if args.op == "swiglu":
                    gate, up = output.split(result_rows, dim=1)
                    output = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
                elif args.op == "add":
                    output = (output.float() + residual_device[:width]).to(torch.bfloat16)
                end.record()
                end.synchronize()
                if args.profile and iteration == args.warmup:
                    torch.cuda.cudart().cudaProfilerStop()
                if iteration >= args.warmup:
                    durations.append(start.elapsed_time(end) * 1000)
            point["marlin_us"].append(statistics.median(durations))
            actual = output[:, rows.tolist()].float().cpu().numpy()
            point["marlin_oracle"] = check_output(actual, reference)

        run_ninfer(0)
        run_marlin()
        run_marlin()
        run_ninfer(1)
        point["marlin_speedup"] = statistics.mean(point["ninfer_us"]) / statistics.mean(point["marlin_us"])
        report["points"].append(point)
        print(json.dumps(point), flush=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if not point["ninfer_oracle"]["passed"] or not point["marlin_oracle"]["passed"]:
            raise RuntimeError("a route failed the independent mathematical oracle")


if __name__ == "__main__":
    main()