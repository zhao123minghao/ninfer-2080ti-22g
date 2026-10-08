#include "artifact/reader.h"
#include "core/device.h"
#include "core/arena.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::int32_t kRows = 256;
constexpr std::int32_t kColumns = 256;
constexpr std::int32_t kTokens = 5;
constexpr std::size_t kCodeBytes = static_cast<std::size_t>(kRows) * kColumns;
constexpr std::size_t kScaleOffset = kCodeBytes;
constexpr std::size_t kScaleBytes = 8;
// 17 distinct E4M3FN codes (a prime count) so a 64-row offset in the row index never aliases the
// code plane: with 4 codes, row and row+64 produced identical weights and hid an off-by-row0 bug.
constexpr std::array<std::uint8_t, 17> kCodes = {0x28, 0x2a, 0x2c, 0x30, 0x32, 0x34, 0x38, 0x3a,
                                                 0x3c, 0x40, 0xa8, 0xaa, 0xac, 0xb0, 0xb2, 0xb4,
                                                 0xb8};
constexpr std::array<std::uint16_t, 4> kScales = {0x3f80, 0x4000, 0x3f00, 0x3fc0};
constexpr std::int32_t kRegisteredRows = 5120;
constexpr std::int32_t kRegisteredColumns = 6144;
constexpr std::int32_t kSwiGluRows = 34816;
constexpr std::int32_t kSwiGluColumns = 5120;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint16_t float_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

double bf16_to_double(std::uint16_t value) {
    return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U));
}

double e4m3fn_to_double(std::uint8_t value) {
    const int sign = (value & 0x80U) == 0 ? 1 : -1;
    const int exponent = (value >> 3U) & 0x0fU;
    const int mantissa = value & 0x07U;
    if (exponent == 0) { return sign * std::ldexp(static_cast<double>(mantissa), -9); }
    if (exponent == 15 && mantissa == 7) {
        throw std::runtime_error("test generated an E4M3FN NaN code");
    }
    return sign * std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0, exponent - 7);
}

std::size_t scale_plane_offset(std::int32_t rows, std::int32_t columns) {
    const std::size_t code_bytes = static_cast<std::size_t>(rows) * columns;
    return (code_bytes + 255U) & ~std::size_t{255U};
}

std::vector<std::byte> make_block_payload(std::int32_t rows, std::int32_t columns) {
    const std::size_t code_bytes = static_cast<std::size_t>(rows) * columns;
    const std::size_t scale_offset = scale_plane_offset(rows, columns);
    const std::size_t m_tiles = static_cast<std::size_t>(rows / 128);
    const std::size_t k_tiles = static_cast<std::size_t>(columns / 128);
    std::vector<std::byte> payload(scale_offset + m_tiles * k_tiles * sizeof(std::uint16_t));
    auto* codes = reinterpret_cast<std::uint8_t*>(payload.data());
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * columns + column;
            codes[index] = kCodes[(static_cast<std::size_t>(row) * 3U +
                                   static_cast<std::size_t>(column) * 5U) %
                                  kCodes.size()];
        }
    }
    for (std::size_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
        for (std::size_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
            const std::uint16_t scale = kScales[(m_tile * 5U + k_tile * 3U) % kScales.size()];
            const std::size_t index = m_tile * k_tiles + k_tile;
            std::memcpy(payload.data() + scale_offset + index * sizeof(scale), &scale,
                        sizeof(scale));
        }
    }
    return payload;
}

ninfer::Weight make_block_weight(void* device_payload, std::size_t payload_bytes,
                                 std::int32_t rows, std::int32_t columns) {
    ninfer::Weight weight{};
    weight.payload = device_payload;
    weight.payload_bytes = payload_bytes;
    weight.qtype = ninfer::QType::FP8_E4M3FN_BLOCK128_BF16S;
    weight.layout = ninfer::QuantLayout::BlockScaleM128K128;
    weight.scale_dtype = ninfer::DType::BF16;
    weight.group = weight.group_size = 128;
    weight.ndim = 2;
    weight.n = rows;
    weight.k = columns;
    weight.shape[0] = weight.padded_shape[0] = rows;
    weight.shape[1] = weight.padded_shape[1] = columns;
    weight.qdata = device_payload;
    weight.scales = static_cast<const std::byte*>(device_payload) +
                    scale_plane_offset(rows, columns);
    weight.scale_ne[0] = columns / 128;
    weight.scale_ne[1] = rows / 128;
    weight.scale_nb[0] = sizeof(std::uint16_t);
    weight.scale_nb[1] = static_cast<std::int64_t>(columns / 128) * sizeof(std::uint16_t);
    weight.scale_nb[2] = weight.scale_nb[1] * (rows / 128);
    weight.scale_nb[3] = weight.scale_nb[2];
    return weight;
}

std::vector<std::byte> make_marlin_block_payload(const std::vector<std::byte>& row_major,
                                                 std::int32_t rows, std::int32_t columns) {
    const std::size_t code_bytes = static_cast<std::size_t>(rows) * columns;
    const std::size_t scale_offset = scale_plane_offset(rows, columns);
    std::vector<std::byte> payload(scale_offset +
                                   static_cast<std::size_t>(rows / 128) * (columns / 128) * 2);
    const auto* source_codes = reinterpret_cast<const std::uint8_t*>(row_major.data());
    auto* codes = reinterpret_cast<std::uint8_t*>(payload.data());
    const int n_tiles = rows / 32;
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const int local_n = row & 31;
            const int local_k = column & 31;
            const int thread = 4 * (local_n & 7) + ((local_k & 15) >> 2);
            const int warp = 2 * (local_n >> 4) + ((local_n >> 3) & 1);
            const int result = local_k >> 4;
            const std::size_t destination =
                static_cast<std::size_t>(((((column / 32) * n_tiles + row / 32) * 32 + thread) *
                                           4 + warp) *
                                          8 + result * 4 + (local_k & 3));
            codes[destination] = source_codes[static_cast<std::size_t>(row) * columns + column];
        }
    }

    // The scale plane is the logical `[N/128, K/128]` plane: the Marlin leaves read one multiplier
    // per row block and K group, so the persistent form stores each of them once.
    const std::size_t source_scale_offset = scale_plane_offset(rows, columns);
    const std::size_t scale_bytes =
        static_cast<std::size_t>(rows / 128) * (columns / 128) * sizeof(std::uint16_t);
    std::memcpy(payload.data() + scale_offset, row_major.data() + source_scale_offset,
                scale_bytes);
    return payload;
}

// Packed 16x32-layout codes for the register-dequantized HMMA prototype. One 16-row x 32-k unit
// holds each lane's 16 codes for two consecutive k-steps contiguously, so one LDG.128 per lane
// feeds two m16k8 A fragments. The scale plane stays the logical [N/128, K/128] plane.
std::vector<std::byte> make_packed_block_payload(const std::vector<std::byte>& row_major,
                                                 std::int32_t rows, std::int32_t columns) {
    const std::size_t scale_offset = scale_plane_offset(rows, columns);
    std::vector<std::byte> payload(scale_offset +
                                   static_cast<std::size_t>(rows / 128) * (columns / 128) * 2);
    const auto* source = reinterpret_cast<const std::uint8_t*>(row_major.data());
    auto* codes = reinterpret_cast<std::uint8_t*>(payload.data());
    const int units_per_row = columns / 32;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const int unit_row = row / 16;
            const int unit_k = column / 32;
            const int local_row = row & 15;
            const int local_k32 = column & 31;
            const int kstep = local_k32 >> 4;
            const int local_k = local_k32 & 15;
            const int lane = (local_row & 7) * 4 + ((local_k & 7) >> 1);
            const int j = 2 * (local_row >> 3) + 4 * (local_k >> 3) + (local_k & 1);
            const std::size_t destination =
                static_cast<std::size_t>(unit_row * units_per_row + unit_k) * 512 +
                lane * 16 + kstep * 8 + j;
            codes[destination] =
                source[static_cast<std::size_t>(row) * columns + column];
        }
    }
    const std::size_t source_scale_offset = scale_plane_offset(rows, columns);
    const std::size_t scale_bytes =
        static_cast<std::size_t>(rows / 128) * (columns / 128) * sizeof(std::uint16_t);
    std::memcpy(payload.data() + scale_offset, row_major.data() + source_scale_offset,
                scale_bytes);
    return payload;
}

ninfer::Weight make_marlin_block_weight(void* device_payload, std::size_t payload_bytes,
                                        std::int32_t rows, std::int32_t columns) {
    ninfer::Weight weight{};
    weight.payload = device_payload;
    weight.payload_bytes = payload_bytes;
    weight.qtype = ninfer::QType::FP8_E4M3FN_BLOCK128_BF16S;
    weight.layout = ninfer::QuantLayout::MarlinFp8Block128;
    weight.scale_dtype = ninfer::DType::BF16;
    weight.group = weight.group_size = 128;
    weight.ndim = 2;
    weight.n = rows;
    weight.k = columns;
    weight.shape[0] = weight.padded_shape[0] = rows;
    weight.shape[1] = weight.padded_shape[1] = columns;
    weight.qdata = device_payload;
    weight.scales = static_cast<const std::byte*>(device_payload) +
                    scale_plane_offset(rows, columns);
    weight.scale_ne[0] = columns / 128;
    weight.scale_ne[1] = rows / 128;
    weight.scale_nb[0] = sizeof(std::uint16_t);
    weight.scale_nb[1] = static_cast<std::int64_t>(columns / 128) * sizeof(std::uint16_t);
    weight.scale_nb[2] = weight.scale_nb[1] * (rows / 128);
    weight.scale_nb[3] = weight.scale_nb[2];
    return weight;
}

ninfer::Weight block_row_view(const ninfer::Weight& parent, std::int32_t row_begin,
                             std::int32_t row_count) {
    ninfer::Weight view = parent;
    const std::int32_t k_tiles = parent.k / 128;
    view.qdata = static_cast<const std::byte*>(parent.qdata) +
                 static_cast<std::uint64_t>(row_begin) * parent.k;
    view.scales = static_cast<const std::byte*>(parent.scales) +
                  static_cast<std::uint64_t>(row_begin / 128) * k_tiles * sizeof(std::uint16_t);
    view.n = view.shape[0] = view.padded_shape[0] = row_count;
    view.scale_ne[1] = row_count / 128;
    view.scale_nb[2] = view.scale_nb[1] * view.scale_ne[1];
    view.scale_nb[3] = view.scale_nb[2];
    return view;
}

double block_linear_fp64(const std::vector<std::byte>& payload,
                         const std::vector<std::uint16_t>& input, std::int32_t row,
                         std::int32_t token, std::int32_t rows, std::int32_t columns,
                         std::int32_t intermediate_rows = 0) {
    const auto* codes = reinterpret_cast<const std::uint8_t*>(payload.data());
    const std::size_t scale_offset = scale_plane_offset(rows, columns);
    const std::int32_t weight_row = row + intermediate_rows;
    const std::int32_t k_tiles = columns / 128;
    double result = 0.0;
    for (std::int32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        double partial = 0.0;
        for (std::int32_t local = 0; local < 128; ++local) {
            const std::int32_t column = k_tile * 128 + local;
            const std::uint8_t code =
                codes[static_cast<std::size_t>(weight_row) * columns + column];
            const std::uint16_t activation =
                input[static_cast<std::size_t>(token) * columns + column];
            partial += e4m3fn_to_double(code) * bf16_to_double(activation);
        }
        std::uint16_t scale = 0;
        const std::size_t scale_index =
            static_cast<std::size_t>(weight_row / 128) * k_tiles + k_tile;
        std::memcpy(&scale, payload.data() + scale_offset + scale_index * sizeof(scale),
                    sizeof(scale));
        result += partial * bf16_to_double(scale);
    }
    return result;
}

struct DeviceAllocation {
    void* pointer = nullptr;

    explicit DeviceAllocation(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&pointer, bytes)); }
    ~DeviceAllocation() {
        if (pointer != nullptr) { (void)cudaFree(pointer); }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
};

void run_block_fp8_linear_oracle(std::int32_t tokens) {
    std::vector<std::byte> payload(kScaleOffset + kScaleBytes);
    auto* code_words = reinterpret_cast<std::uint8_t*>(payload.data());
    for (std::size_t index = 0; index < kCodeBytes; ++index) {
        code_words[index] = kCodes[(index + index / kColumns) % kCodes.size()];
    }
    std::memcpy(payload.data() + kScaleOffset, kScales.data(), kScaleBytes);

    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * tokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 17U) % 31U) - 15) / 32.0F;
        input[index] = float_to_bf16(value);
    }
    std::vector<std::uint16_t> output(static_cast<std::size_t>(kRows) * tokens);

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_output(output.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));

    ninfer::Weight weight{};
    weight.payload = device_payload.pointer;
    weight.payload_bytes = payload.size();
    weight.qtype = ninfer::QType::FP8_E4M3FN_BLOCK128_BF16S;
    weight.layout = ninfer::QuantLayout::BlockScaleM128K128;
    weight.scale_dtype = ninfer::DType::BF16;
    weight.group = 128;
    weight.group_size = 128;
    weight.ndim = 2;
    weight.n = kRows;
    weight.k = kColumns;
    weight.shape[0] = weight.padded_shape[0] = kRows;
    weight.shape[1] = weight.padded_shape[1] = kColumns;
    weight.qdata = device_payload.pointer;
    weight.scales = static_cast<const std::byte*>(device_payload.pointer) + kScaleOffset;
    weight.scale_ne[0] = kColumns / 128;
    weight.scale_ne[1] = kRows / 128;
    weight.scale_nb[0] = 2;
    weight.scale_nb[1] = (kColumns / 128) * 2;
    weight.scale_nb[2] = weight.scale_nb[1] * (kRows / 128);
    weight.scale_nb[3] = weight.scale_nb[2];

    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, tokens});
    ninfer::Tensor y(device_output.pointer, ninfer::DType::BF16, {kRows, tokens});
    ninfer::ops::linear(x, weight, y, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.pointer,
                          output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    double maximum_error = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < kRows; ++row) {
            double expected = 0.0;
            for (int k_tile = 0; k_tile < kColumns / 128; ++k_tile) {
                double partial = 0.0;
                for (int local = 0; local < 128; ++local) {
                    const int column = k_tile * 128 + local;
                    const std::uint8_t code = code_words[static_cast<std::size_t>(row) * kColumns +
                                                          column];
                    const std::uint16_t activation =
                        input[static_cast<std::size_t>(token) * kColumns + column];
                    partial += e4m3fn_to_double(code) * bf16_to_double(activation);
                }
                const std::uint16_t scale =
                    kScales[(row / 128) * (kColumns / 128) + k_tile];
                expected += partial * bf16_to_double(scale);
            }
            const double actual = bf16_to_double(output[static_cast<std::size_t>(token) * kRows +
                                                         row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            // The staged HMMA leaf accumulates in fp16 over the full reduction (matching the
            // external Marlin kernel), so its profile error is ~1% relative rather than the
            // fold-to-fp32 path's sub-0.5%. The criterion is set to that profile.
            require(error <= 0.02 + 0.02 * std::abs(expected),
                    ("block-FP8 Linear differs from the FP64 block-scale oracle (tokens=" +
                     std::to_string(tokens) + ')')
                        .c_str());
        }
    }
    std::cout << "block-FP8 Linear oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_block_fp8_linear_broadcast_oracle(std::int32_t tokens) {
    // The broadcast kernel now consumes the packed 16x32 layout (block-level shared stage of the
    // raw packed codes, ALU decode in registers), so it reads the same packed payload as the
    // packed kernel; the FP64 oracle still runs over the row-major form of the same logical
    // weights.
    const std::vector<std::byte> row_major = make_block_payload(kRows, kColumns);
    const std::vector<std::byte> packed = make_packed_block_payload(row_major, kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * tokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 17U) % 31U) - 15) / 32.0F;
        input[index] = float_to_bf16(value);
    }
    std::vector<std::uint16_t> output(static_cast<std::size_t>(kRows) * tokens);

    DeviceAllocation device_payload(packed.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_output(output.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, packed.data(), packed.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));

    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, packed.size(), kRows, kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, tokens});
    ninfer::Tensor y(device_output.pointer, ninfer::DType::BF16, {kRows, tokens});
    setenv("NINFER_FP8_BLOCK_LINEAR_BROADCAST", "1", 1);
    setenv("NINFER_FP8_BLOCK_ROUTE", "hmma", 1);
    ninfer::ops::linear(x, weight, y, nullptr);
    unsetenv("NINFER_FP8_BLOCK_LINEAR_BROADCAST");
    unsetenv("NINFER_FP8_BLOCK_ROUTE");
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.pointer,
                          output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    double maximum_error = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < kRows; ++row) {
            const double expected =
                block_linear_fp64(row_major, input, row, token, kRows, kColumns);
            const double actual = bf16_to_double(
                output[static_cast<std::size_t>(token) * kRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    ("broadcast block-FP8 Linear differs from the FP64 block-scale oracle "
                     "(tokens=" +
                     std::to_string(tokens) + ')')
                        .c_str());
        }
    }
    std::cout << "broadcast block-FP8 Linear oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_block_fp8_linear_packed_oracle(std::int32_t tokens) {
    const std::vector<std::byte> row_major = make_block_payload(kRows, kColumns);
    const std::vector<std::byte> packed = make_packed_block_payload(row_major, kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * tokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 17U) % 31U) - 15) / 32.0F;
        input[index] = float_to_bf16(value);
    }
    std::vector<std::uint16_t> output(static_cast<std::size_t>(kRows) * tokens);

    DeviceAllocation device_payload(packed.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_output(output.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, packed.data(), packed.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));

    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, packed.size(), kRows, kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, tokens});
    ninfer::Tensor y(device_output.pointer, ninfer::DType::BF16, {kRows, tokens});
    setenv("NINFER_FP8_BLOCK_LINEAR_PACKED", "1", 1);
    setenv("NINFER_FP8_BLOCK_ROUTE", "hmma", 1);
    ninfer::ops::linear(x, weight, y, nullptr);
    unsetenv("NINFER_FP8_BLOCK_LINEAR_PACKED");
    unsetenv("NINFER_FP8_BLOCK_ROUTE");
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.pointer,
                          output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    double maximum_error = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < kRows; ++row) {
            const double expected =
                block_linear_fp64(row_major, input, row, token, kRows, kColumns);
            const double actual = bf16_to_double(
                output[static_cast<std::size_t>(token) * kRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    ("packed block-FP8 Linear differs from the FP64 block-scale oracle (tokens=" +
                     std::to_string(tokens) + ')')
                        .c_str());
        }
    }
    std::cout << "packed block-FP8 Linear oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_block_fp8_linear_packed256_oracle(std::int32_t tokens) {
    const std::vector<std::byte> row_major = make_block_payload(kRows, kColumns);
    const std::vector<std::byte> packed = make_packed_block_payload(row_major, kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * tokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 17U) % 31U) - 15) / 32.0F;
        input[index] = float_to_bf16(value);
    }
    std::vector<std::uint16_t> output(static_cast<std::size_t>(kRows) * tokens);

    DeviceAllocation device_payload(packed.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_output(output.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, packed.data(), packed.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));

    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, packed.size(), kRows, kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, tokens});
    ninfer::Tensor y(device_output.pointer, ninfer::DType::BF16, {kRows, tokens});
    setenv("NINFER_FP8_BLOCK_LINEAR_PACKED256", "1", 1);
    setenv("NINFER_FP8_BLOCK_ROUTE", "hmma", 1);
    ninfer::ops::linear(x, weight, y, nullptr);
    unsetenv("NINFER_FP8_BLOCK_LINEAR_PACKED256");
    unsetenv("NINFER_FP8_BLOCK_ROUTE");
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.pointer,
                          output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    double maximum_error = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < kRows; ++row) {
            const double expected =
                block_linear_fp64(row_major, input, row, token, kRows, kColumns);
            const double actual = bf16_to_double(
                output[static_cast<std::size_t>(token) * kRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    ("packed256 block-FP8 Linear differs from the FP64 block-scale oracle "
                     "(tokens=" +
                     std::to_string(tokens) + ')')
                        .c_str());
        }
    }
    std::cout << "packed256 block-FP8 Linear oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_marlin_block_fp8_linear_oracle(std::int32_t tokens) {
    constexpr std::int32_t kMarlinRows = 256;
    constexpr std::int32_t kMarlinColumns = 256;
    const std::int32_t kMarlinTokens = tokens;
    const std::vector<std::byte> row_major = make_block_payload(kMarlinRows, kMarlinColumns);
    const std::vector<std::byte> payload =
        make_marlin_block_payload(row_major, kMarlinRows, kMarlinColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kMarlinColumns) * kMarlinTokens);
    std::vector<std::uint16_t> output(static_cast<std::size_t>(kMarlinRows) * kMarlinTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 31U) % 47U) - 23) / 128.0F;
        input[index] = float_to_bf16(value);
    }

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_output(output.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    const ninfer::Weight weight =
        make_marlin_block_weight(device_payload.pointer, payload.size(), kMarlinRows, kMarlinColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16,
                           {kMarlinColumns, kMarlinTokens});
    ninfer::Tensor out(device_output.pointer, ninfer::DType::BF16, {kMarlinRows, kMarlinTokens});
    ninfer::ops::linear(x, weight, out, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.pointer, output.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));

    double maximum_error = 0.0;
    for (int token = 0; token < kMarlinTokens; ++token) {
        for (int row = 0; row < kMarlinRows; ++row) {
            const double expected = block_linear_fp64(row_major, input, row, token, kMarlinRows,
                                                       kMarlinColumns);
            const double actual =
                bf16_to_double(output[static_cast<std::size_t>(token) * kMarlinRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    "Marlin block-FP8 Linear differs from the independent FP64 oracle");
        }
    }
    std::cout << "Marlin block-FP8 Linear oracle (T=" << kMarlinTokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_marlin_block_fp8_swiglu_oracle(std::int32_t tokens) {
    const std::int32_t kTokens = tokens;
    constexpr std::int32_t kIntermediateRows = kSwiGluRows / 2;
    const std::vector<std::byte> row_major = make_block_payload(kSwiGluRows, kSwiGluColumns);
    const std::vector<std::byte> payload =
        make_marlin_block_payload(row_major, kSwiGluRows, kSwiGluColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kSwiGluColumns) * kTokens);
    std::vector<std::uint16_t> output(static_cast<std::size_t>(kIntermediateRows) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 17U) % 31U) - 15) / 128.0F;
        input[index] = float_to_bf16(value);
    }

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_output(output.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    const ninfer::Weight weight =
        make_marlin_block_weight(device_payload.pointer, payload.size(), kSwiGluRows, kSwiGluColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16,
                           {kSwiGluColumns, kTokens});
    ninfer::Tensor result(device_output.pointer, ninfer::DType::BF16,
                          {kIntermediateRows, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::linear_swiglu(x, weight, result, ninfer::ops::LinearPolicy::A16Only, workspace,
                               nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.pointer,
                          output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    constexpr std::array<int, 8> kRowsToCheck = {0, 63, 64, 127, 128, 8703, 8704, 17407};
    double maximum_error = 0.0;
    for (const int token : {0, kTokens / 2, kTokens - 1}) {
        for (const int row : kRowsToCheck) {
            const double gate =
                block_linear_fp64(row_major, input, row, token, kSwiGluRows, kSwiGluColumns);
            const double up = block_linear_fp64(row_major, input, row, token, kSwiGluRows,
                                                kSwiGluColumns, kIntermediateRows);
            const double expected = gate / (1.0 + std::exp(-gate)) * up;
            const double actual =
                bf16_to_double(output[static_cast<std::size_t>(token) * kIntermediateRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    "Marlin block-FP8 SwiGLU differs from the independent FP64 oracle");
        }
    }
    std::cout << "Marlin block-FP8 SwiGLU oracle (T=" << kTokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_marlin_block_fp8_linear_add_oracle() {
    constexpr std::int32_t kRows = 5120;
    constexpr std::int32_t kColumns = 6144;
    constexpr std::int32_t kTokens = 4;
    const std::vector<std::byte> row_major = make_block_payload(kRows, kColumns);
    const std::vector<std::byte> payload = make_marlin_block_payload(row_major, kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * kTokens);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(kRows) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 11U) % 29U) - 14) / 64.0F;
        input[index] = float_to_bf16(value);
    }
    for (std::size_t index = 0; index < residual.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 7U) % 19U) - 9) / 128.0F;
        residual[index] = float_to_bf16(value);
    }
    const std::vector<std::uint16_t> initial_residual = residual;

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_residual(residual.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_residual.pointer, residual.data(),
                          residual.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    const ninfer::Weight weight =
        make_marlin_block_weight(device_payload.pointer, payload.size(), kRows, kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, kTokens});
    ninfer::Tensor result(device_residual.pointer, ninfer::DType::BF16, {kRows, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::linear_add(x, weight, result, ninfer::ops::LinearPolicy::A16Only, workspace,
                            nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(residual.data(), device_residual.pointer,
                          residual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    constexpr std::array<int, 6> kRowsToCheck = {0, 127, 128, 2559, 2560, 5119};
    double maximum_error = 0.0;
    for (int token = 0; token < kTokens; ++token) {
        for (const int row : kRowsToCheck) {
            const double expected =
                block_linear_fp64(row_major, input, row, token, kRows, kColumns) +
                bf16_to_double(initial_residual[static_cast<std::size_t>(token) * kRows + row]);
            const double actual =
                bf16_to_double(residual[static_cast<std::size_t>(token) * kRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    "Marlin block-FP8 LinearAdd differs from the independent FP64 oracle");
        }
    }
    std::cout << "Marlin block-FP8 LinearAdd oracle: passed; max_abs_error=" << maximum_error
              << '\n';
}

void run_marlin_block_fp8_attention_oracle() {
    constexpr std::int32_t kRows = 14336;
    constexpr std::int32_t kColumns = 5120;
    constexpr std::int32_t kTokens = 1;
    const std::vector<std::byte> row_major = make_block_payload(kRows, kColumns);
    const std::vector<std::byte> payload = make_marlin_block_payload(row_major, kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * kTokens);
    std::array<std::vector<std::uint16_t>, 4> outputs{
        std::vector<std::uint16_t>(6144U * kTokens), std::vector<std::uint16_t>(6144U * kTokens),
        std::vector<std::uint16_t>(1024U * kTokens), std::vector<std::uint16_t>(1024U * kTokens)};
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 19U) % 37U) - 18) / 256.0F;
        input[index] = float_to_bf16(value);
    }

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    std::array<DeviceAllocation, 4> device_outputs{
        DeviceAllocation(outputs[0].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[1].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[2].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[3].size() * sizeof(std::uint16_t))};
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    const ninfer::Weight weight =
        make_marlin_block_weight(device_payload.pointer, payload.size(), kRows, kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, kTokens});
    ninfer::Tensor query(device_outputs[0].pointer, ninfer::DType::BF16, {6144, kTokens});
    ninfer::Tensor gate(device_outputs[1].pointer, ninfer::DType::BF16, {6144, kTokens});
    ninfer::Tensor key(device_outputs[2].pointer, ninfer::DType::BF16, {1024, kTokens});
    ninfer::Tensor value(device_outputs[3].pointer, ninfer::DType::BF16, {1024, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::attn_input_proj(x, weight, query, gate, key, value,
                                 ninfer::ops::LinearPolicy::A16Only, workspace, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t output = 0; output < outputs.size(); ++output) {
        CUDA_CHECK(cudaMemcpy(outputs[output].data(), device_outputs[output].pointer,
                              outputs[output].size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
    }

    constexpr std::array<int, 9> kRowsToCheck = {0, 127, 6143, 6144, 7167, 7168, 13311, 13312,
                                                  14335};
    double maximum_error = 0.0;
    for (const int row : kRowsToCheck) {
        std::size_t output_index = 0;
        int output_row = 0;
        int output_rows = 0;
        if (row < 6144) {
            output_index = 0;
            output_row = row;
            output_rows = 6144;
        } else if (row < 7168) {
            output_index = 2;
            output_row = row - 6144;
            output_rows = 1024;
        } else if (row < 13312) {
            output_index = 1;
            output_row = row - 7168;
            output_rows = 6144;
        } else {
            output_index = 3;
            output_row = row - 13312;
            output_rows = 1024;
        }
        const double expected = block_linear_fp64(row_major, input, row, 0, kRows, kColumns);
        const double actual =
            bf16_to_double(outputs[output_index][static_cast<std::size_t>(output_row)]);
        const double error = std::abs(actual - expected);
        maximum_error = std::max(maximum_error, error);
        require(error <= 0.005 + 0.004 * std::abs(expected),
                "Marlin block-FP8 attention projection differs from the independent FP64 oracle");
    }
    std::cout << "Marlin block-FP8 attention oracle: passed; max_abs_error=" << maximum_error
              << '\n';
}

void run_block_fp8_linear_add_oracle(std::int32_t tokens) {
    const std::int32_t kTokens = tokens;
    const std::vector<std::byte> payload =
        make_block_payload(kRegisteredRows, kRegisteredColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kRegisteredColumns) * kTokens);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(kRegisteredRows) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 11U) % 29U) - 14) / 64.0F;
        input[index] = float_to_bf16(value);
    }
    for (std::size_t index = 0; index < residual.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 7U) % 19U) - 9) / 128.0F;
        residual[index] = float_to_bf16(value);
    }
    const std::vector<std::uint16_t> initial_residual = residual;

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_residual(residual.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_residual.pointer, residual.data(),
                          residual.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, payload.size(), kRegisteredRows,
                          kRegisteredColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16,
                           {kRegisteredColumns, kTokens});
    ninfer::Tensor result(device_residual.pointer, ninfer::DType::BF16,
                          {kRegisteredRows, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::linear_add(x, weight, result, ninfer::ops::LinearPolicy::A16Only, workspace,
                            nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(residual.data(), device_residual.pointer,
                          residual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    constexpr std::array<int, 6> kRowsToCheck = {0, 127, 128, 2559, 2560, 5119};
    const std::array<int, 3> kTokensToCheck = {0, tokens / 2, tokens - 1};
    double maximum_error = 0.0;
    for (const int token : kTokensToCheck) {
        for (const int row : kRowsToCheck) {
            const double expected =
                block_linear_fp64(payload, input, row, token, kRegisteredRows,
                                  kRegisteredColumns) +
                bf16_to_double(initial_residual[static_cast<std::size_t>(token) * kRegisteredRows +
                                                row]);
            const double actual = bf16_to_double(
                residual[static_cast<std::size_t>(token) * kRegisteredRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            // fp16 accumulation profile (see the Linear oracle above).
            require(error <= 0.02 + 0.02 * std::abs(expected),
                    "block-FP8 LinearAdd differs from the FP64 block-scale oracle");
        }
    }
    std::cout << "block-FP8 LinearAdd oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_block_fp8_swiglu_oracle(std::int32_t tokens) {
    const std::int32_t kTokens = tokens;
    constexpr std::int32_t kIntermediateRows = kSwiGluRows / 2;
    const std::vector<std::byte> payload = make_block_payload(kSwiGluRows, kSwiGluColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kSwiGluColumns) * kTokens);
    std::vector<std::uint16_t> output(static_cast<std::size_t>(kIntermediateRows) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 17U) % 31U) - 15) / 128.0F;
        input[index] = float_to_bf16(value);
    }

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_output(output.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));

    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, payload.size(), kSwiGluRows, kSwiGluColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kSwiGluColumns, kTokens});
    ninfer::Tensor result(device_output.pointer, ninfer::DType::BF16,
                          {kIntermediateRows, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::linear_swiglu(x, weight, result, ninfer::ops::LinearPolicy::A16Only, workspace,
                               nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.pointer,
                          output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    constexpr std::array<int, 10> kRowsToCheck = {0, 1, 127, 128, 255, 4096, 8703, 8704, 16383,
                                                  17407};
    const std::array<int, 5> kTokensToCheck = {
        0, tokens / 2, std::min(127, tokens - 1), std::min(128, tokens - 1), tokens - 1};
    double maximum_error = 0.0;
    for (const int token : kTokensToCheck) {
        for (const int row : kRowsToCheck) {
            const double gate = block_linear_fp64(payload, input, row, token, kSwiGluRows,
                                                  kSwiGluColumns);
            const double up = block_linear_fp64(payload, input, row, token, kSwiGluRows,
                                                kSwiGluColumns, kIntermediateRows);
            const double expected = gate / (1.0 + std::exp(-gate)) * up;
            const double actual = bf16_to_double(
                output[static_cast<std::size_t>(token) * kIntermediateRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            // fp16 accumulation profile (see the Linear oracle above); the fused silu(gate)*up
            // epilogue multiplies two fp16-accumulated dots, so the profile error is larger here.
            require(error <= 0.1 + 0.05 * std::abs(expected),
                    "block-FP8 SwiGLU differs from the FP64 block-scale oracle");
        }
    }
    std::cout << "block-FP8 SwiGLU oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_block_fp8_swiglu_packed_oracle(std::int32_t tokens) {
    const std::int32_t kTokens = tokens;
    constexpr std::int32_t kIntermediateRows = kSwiGluRows / 2;
    const std::vector<std::byte> row_major = make_block_payload(kSwiGluRows, kSwiGluColumns);
    const std::vector<std::byte> packed =
        make_packed_block_payload(row_major, kSwiGluRows, kSwiGluColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kSwiGluColumns) * kTokens);
    std::vector<std::uint16_t> output(static_cast<std::size_t>(kIntermediateRows) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 17U) % 31U) - 15) / 128.0F;
        input[index] = float_to_bf16(value);
    }

    DeviceAllocation device_payload(packed.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_output(output.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, packed.data(), packed.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));

    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, packed.size(), kSwiGluRows, kSwiGluColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kSwiGluColumns, kTokens});
    ninfer::Tensor result(device_output.pointer, ninfer::DType::BF16,
                          {kIntermediateRows, kTokens});
    ninfer::WorkspaceArena workspace(256);
    setenv("NINFER_FP8_BLOCK_PACKED", "1", 1);
    setenv("NINFER_FP8_BLOCK_ROUTE", "hmma", 1);
    ninfer::ops::linear_swiglu(x, weight, result, ninfer::ops::LinearPolicy::A16Only, workspace,
                               nullptr);
    unsetenv("NINFER_FP8_BLOCK_PACKED");
    unsetenv("NINFER_FP8_BLOCK_ROUTE");
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.pointer,
                          output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    constexpr std::array<int, 10> kRowsToCheck = {0, 1, 127, 128, 255, 4096, 8703, 8704, 16383,
                                                  17407};
    const std::array<int, 5> kTokensToCheck = {
        0, tokens / 2, std::min(127, tokens - 1), std::min(128, tokens - 1), tokens - 1};
    double maximum_error = 0.0;
    for (const int token : kTokensToCheck) {
        for (const int row : kRowsToCheck) {
            const double gate =
                block_linear_fp64(row_major, input, row, token, kSwiGluRows, kSwiGluColumns);
            const double up = block_linear_fp64(row_major, input, row, token, kSwiGluRows,
                                                kSwiGluColumns, kIntermediateRows);
            const double expected = gate / (1.0 + std::exp(-gate)) * up;
            const double actual = bf16_to_double(
                output[static_cast<std::size_t>(token) * kIntermediateRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    "packed block-FP8 SwiGLU differs from the FP64 block-scale oracle");
        }
    }
    std::cout << "packed block-FP8 SwiGLU oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_block_fp8_attention_oracle(std::int32_t tokens) {
    constexpr std::int32_t kRows = 14336;
    constexpr std::int32_t kColumns = 5120;
    const std::int32_t kTokens = tokens;
    constexpr std::int32_t kQueryRows = 6144;
    constexpr std::int32_t kKeyValueRows = 1024;
    constexpr std::int32_t kKeyBegin = kQueryRows;
    constexpr std::int32_t kGateBegin = kQueryRows + kKeyValueRows;
    constexpr std::int32_t kValueBegin = kGateBegin + kQueryRows;
    const std::vector<std::byte> payload = make_block_payload(kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 19U) % 37U) - 18) / 256.0F;
        input[index] = float_to_bf16(value);
    }
    std::array<std::vector<std::uint16_t>, 4> outputs{
        std::vector<std::uint16_t>(static_cast<std::size_t>(kQueryRows) * kTokens),
        std::vector<std::uint16_t>(static_cast<std::size_t>(kQueryRows) * kTokens),
        std::vector<std::uint16_t>(static_cast<std::size_t>(kKeyValueRows) * kTokens),
        std::vector<std::uint16_t>(static_cast<std::size_t>(kKeyValueRows) * kTokens)};
    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    std::array<DeviceAllocation, 4> device_outputs{
        DeviceAllocation(outputs[0].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[1].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[2].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[3].size() * sizeof(std::uint16_t))};
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    const ninfer::Weight weight = make_block_weight(device_payload.pointer, payload.size(), kRows,
                                                     kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, kTokens});
    ninfer::Tensor query(device_outputs[0].pointer, ninfer::DType::BF16,
                         {kQueryRows, kTokens});
    ninfer::Tensor gate(device_outputs[1].pointer, ninfer::DType::BF16,
                        {kQueryRows, kTokens});
    ninfer::Tensor key(device_outputs[2].pointer, ninfer::DType::BF16,
                       {kKeyValueRows, kTokens});
    ninfer::Tensor value(device_outputs[3].pointer, ninfer::DType::BF16,
                         {kKeyValueRows, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::attn_input_proj(x, weight, query, gate, key, value,
                                 ninfer::ops::LinearPolicy::A16Only, workspace, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t output = 0; output < outputs.size(); ++output) {
        CUDA_CHECK(cudaMemcpy(outputs[output].data(), device_outputs[output].pointer,
                              outputs[output].size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
    }

    constexpr std::array<int, 9> kRowsToCheck = {0, 127, 6143, 6144, 7167, 7168, 13311, 13312,
                                                  14335};
    const std::array<int, 5> kTokensToCheck = {
        0, tokens / 2, std::min(127, tokens - 1), std::min(128, tokens - 1), tokens - 1};
    double maximum_error = 0.0;
    for (const int token : kTokensToCheck) {
        for (const int row : kRowsToCheck) {
            std::size_t output_index = 0;
            int output_row = 0;
            if (row < kQueryRows) {
                output_index = 0;
                output_row = row;
            } else if (row < kGateBegin) {
                output_index = 2;
                output_row = row - kKeyBegin;
            } else if (row < kValueBegin) {
                output_index = 1;
                output_row = row - kGateBegin;
            } else {
                output_index = 3;
                output_row = row - kValueBegin;
            }
            const double expected =
                block_linear_fp64(payload, input, row, token, kRows, kColumns);
            const double actual = bf16_to_double(
                outputs[output_index][static_cast<std::size_t>(token) *
                                          outputs[output_index].size() / kTokens +
                                      output_row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            // fp16 accumulation profile (see the Linear oracle above).
            require(error <= 0.02 + 0.02 * std::abs(expected),
                    "block-FP8 attention projection differs from the FP64 block-scale oracle");
        }
    }
    std::cout << "block-FP8 attention oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_block_fp8_gdn_oracle(std::int32_t tokens) {
    constexpr std::int32_t kRows = 16384;
    constexpr std::int32_t kColumns = 5120;
    const std::int32_t kTokens = tokens;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows = kRows - kQkvRows;
    const std::vector<std::byte> payload = make_block_payload(kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * kTokens);
    std::vector<std::uint16_t> qkv(static_cast<std::size_t>(kQkvRows) * kTokens);
    std::vector<std::uint16_t> z(static_cast<std::size_t>(kZRows) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 23U) % 41U) - 20) / 256.0F;
        input[index] = float_to_bf16(value);
    }

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_qkv(qkv.size() * sizeof(std::uint16_t));
    DeviceAllocation device_z(z.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    const ninfer::Weight weight = make_block_weight(device_payload.pointer, payload.size(), kRows,
                                                     kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, kTokens});
    ninfer::Tensor qkv_output(device_qkv.pointer, ninfer::DType::BF16, {kQkvRows, kTokens});
    ninfer::Tensor z_output(device_z.pointer, ninfer::DType::BF16, {kZRows, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::gdn_input_proj(x, weight, qkv_output, z_output,
                                ninfer::ops::LinearPolicy::A16Only, workspace, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(qkv.data(), device_qkv.pointer, qkv.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(z.data(), device_z.pointer, z.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));

    constexpr std::array<int, 5> kRowsToCheck = {0, 127, 10239, 10240, 16383};
    const std::array<int, 5> kTokensToCheck = {
        0, tokens / 2, std::min(127, tokens - 1), std::min(128, tokens - 1), tokens - 1};
    double maximum_error = 0.0;
    for (const int token : kTokensToCheck) {
        for (const int row : kRowsToCheck) {
            const double expected =
                block_linear_fp64(payload, input, row, token, kRows, kColumns);
            const double actual = row < kQkvRows
                                      ? bf16_to_double(qkv[static_cast<std::size_t>(token) *
                                                               kQkvRows +
                                                           row])
                                      : bf16_to_double(z[static_cast<std::size_t>(token) * kZRows +
                                                         row - kQkvRows]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            // fp16 accumulation profile (see the Linear oracle above).
            require(error <= 0.02 + 0.02 * std::abs(expected),
                    "block-FP8 GDN projection differs from the FP64 block-scale oracle");
        }
    }
    std::cout << "block-FP8 GDN oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_block_fp8_linear_pair_oracle() {
    constexpr std::int32_t kRows = 14336;
    constexpr std::int32_t kColumns = 5120;
    constexpr std::int32_t kPairRows = 1024;
    constexpr std::int32_t kKeyBegin = 6144;
    constexpr std::int32_t kValueBegin = 13312;
    constexpr std::int32_t kTokens = 5;
    const std::vector<std::byte> payload = make_block_payload(kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * kTokens);
    std::array<std::vector<std::uint16_t>, 2> outputs{
        std::vector<std::uint16_t>(static_cast<std::size_t>(kPairRows) * kTokens),
        std::vector<std::uint16_t>(static_cast<std::size_t>(kPairRows) * kTokens)};
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 29U) % 43U) - 21) / 256.0F;
        input[index] = float_to_bf16(value);
    }
    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    std::array<DeviceAllocation, 2> device_outputs{
        DeviceAllocation(outputs[0].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[1].size() * sizeof(std::uint16_t))};
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));

    const ninfer::Weight parent =
        make_block_weight(device_payload.pointer, payload.size(), kRows, kColumns);
    const ninfer::Weight key_weight = block_row_view(parent, kKeyBegin, kPairRows);
    const ninfer::Weight value_weight = block_row_view(parent, kValueBegin, kPairRows);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, kTokens});
    ninfer::Tensor key(device_outputs[0].pointer, ninfer::DType::BF16, {kPairRows, kTokens});
    ninfer::Tensor value(device_outputs[1].pointer, ninfer::DType::BF16, {kPairRows, kTokens});
    ninfer::ops::linear_pair(x, key_weight, value_weight, key, value, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t output = 0; output < outputs.size(); ++output) {
        CUDA_CHECK(cudaMemcpy(outputs[output].data(), device_outputs[output].pointer,
                              outputs[output].size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
    }

    constexpr std::array<int, 4> kRowsToCheck = {0, 127, 128, 1023};
    constexpr std::array<int, 3> kTokensToCheck = {0, 2, 4};
    double maximum_error = 0.0;
    for (const int token : kTokensToCheck) {
        for (const int row : kRowsToCheck) {
            const double key_expected = block_linear_fp64(
                payload, input, kKeyBegin + row, token, kRows, kColumns);
            const double value_expected = block_linear_fp64(
                payload, input, kValueBegin + row, token, kRows, kColumns);
            const double key_actual =
                bf16_to_double(outputs[0][static_cast<std::size_t>(token) * kPairRows + row]);
            const double value_actual =
                bf16_to_double(outputs[1][static_cast<std::size_t>(token) * kPairRows + row]);
            const double key_error = std::abs(key_actual - key_expected);
            const double value_error = std::abs(value_actual - value_expected);
            maximum_error = std::max({maximum_error, key_error, value_error});
            require(key_error <= 0.005 + 0.004 * std::abs(key_expected) &&
                        value_error <= 0.005 + 0.004 * std::abs(value_expected),
                    "block-FP8 LinearPair differs from the FP64 block-scale oracle");
        }
    }
    std::cout << "block-FP8 LinearPair oracle: passed; max_abs_error=" << maximum_error << '\n';
}

std::vector<std::byte> real_block_payload(const ninfer::artifact::Reader& reader,
                                          const char* object_name, std::int32_t rows,
                                          std::int32_t columns) {
    const auto* descriptor = reader.find(object_name);
    if (descriptor == nullptr) { throw std::runtime_error("real FP8 object is missing"); }
    const ninfer::artifact::PayloadSpan source_payload = reader.payload(*descriptor);
    const std::size_t expected_bytes = scale_plane_offset(rows, columns) +
        static_cast<std::size_t>(rows / 128) * (columns / 128) * sizeof(std::uint16_t);
    if (source_payload.data.size() != expected_bytes) {
        throw std::runtime_error("real FP8 object has unexpected payload size");
    }
    return {source_payload.data.begin(), source_payload.data.end()};
}

std::vector<std::byte> apply_tensor_slice(std::span<const std::byte> parent,
                                           const ninfer::artifact::TensorSlice& slice) {
    std::vector<std::byte> result(static_cast<std::size_t>(slice.encoded_bytes));
    std::copy(slice.prefix.begin(), slice.prefix.end(), result.begin());
    for (const ninfer::artifact::PlaneCopy& copy : slice.copies) {
        require(copy.source_offset + copy.bytes <= parent.size() &&
                    copy.dest_offset + copy.bytes <= result.size(),
                "real block-FP8 shard copy escapes its payload");
        std::copy_n(parent.begin() + static_cast<std::ptrdiff_t>(copy.source_offset),
                    static_cast<std::ptrdiff_t>(copy.bytes),
                    result.begin() + static_cast<std::ptrdiff_t>(copy.dest_offset));
    }
    return result;
}

std::vector<std::byte> expected_block_row_shard(std::span<const std::byte> parent,
                                                std::int32_t rows, std::int32_t columns,
                                                std::span<const ninfer::artifact::SliceRange> ranges) {
    const std::size_t parent_scale = scale_plane_offset(rows, columns);
    const std::size_t k_tiles = static_cast<std::size_t>(columns / 128);
    std::size_t shard_rows = 0;
    for (const auto& range : ranges) { shard_rows += static_cast<std::size_t>(range.count); }
    const std::size_t shard_scale = scale_plane_offset(static_cast<std::int32_t>(shard_rows), columns);
    std::vector<std::byte> result(shard_scale + (shard_rows / 128) * k_tiles * sizeof(std::uint16_t));
    std::size_t destination_row = 0;
    std::size_t destination_tile = 0;
    for (const auto& range : ranges) {
        for (std::uint64_t row = range.begin; row < range.begin + range.count; ++row) {
            std::copy_n(parent.begin() + static_cast<std::ptrdiff_t>(row * columns), columns,
                        result.begin() + static_cast<std::ptrdiff_t>(destination_row * columns));
            ++destination_row;
        }
        for (std::uint64_t tile = range.begin / 128;
             tile < (range.begin + range.count) / 128; ++tile) {
            for (std::size_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                const std::size_t source_offset = parent_scale +
                    (static_cast<std::size_t>(tile) * k_tiles + k_tile) * sizeof(std::uint16_t);
                const std::size_t destination_offset = shard_scale +
                    (destination_tile * k_tiles + k_tile) * sizeof(std::uint16_t);
                std::copy_n(parent.begin() + static_cast<std::ptrdiff_t>(source_offset),
                            sizeof(std::uint16_t),
                            result.begin() + static_cast<std::ptrdiff_t>(destination_offset));
            }
            ++destination_tile;
        }
    }
    return result;
}

std::vector<std::byte> expected_block_column_shard(std::span<const std::byte> parent,
                                                   std::int32_t rows, std::int32_t columns,
                                                   ninfer::artifact::SliceRange range) {
    const std::size_t parent_scale = scale_plane_offset(rows, columns);
    const std::size_t parent_k_tiles = static_cast<std::size_t>(columns / 128);
    const std::size_t shard_columns = static_cast<std::size_t>(range.count);
    const std::size_t shard_scale = scale_plane_offset(rows, static_cast<std::int32_t>(shard_columns));
    const std::size_t shard_k_tiles = shard_columns / 128;
    std::vector<std::byte> result(shard_scale +
                                  static_cast<std::size_t>(rows / 128) * shard_k_tiles *
                                      sizeof(std::uint16_t));
    for (std::int32_t row = 0; row < rows; ++row) {
        std::copy_n(parent.begin() + static_cast<std::ptrdiff_t>(
                                      static_cast<std::size_t>(row) * columns + range.begin),
                    static_cast<std::ptrdiff_t>(range.count),
                    result.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(row) *
                                                                  shard_columns));
    }
    for (std::size_t m_tile = 0; m_tile < static_cast<std::size_t>(rows / 128); ++m_tile) {
        for (std::size_t k_tile = 0; k_tile < shard_k_tiles; ++k_tile) {
            const std::size_t source_offset = parent_scale +
                (m_tile * parent_k_tiles + range.begin / 128 + k_tile) * sizeof(std::uint16_t);
            const std::size_t destination_offset = shard_scale +
                (m_tile * shard_k_tiles + k_tile) * sizeof(std::uint16_t);
            std::copy_n(parent.begin() + static_cast<std::ptrdiff_t>(source_offset),
                        sizeof(std::uint16_t),
                        result.begin() + static_cast<std::ptrdiff_t>(destination_offset));
        }
    }
    return result;
}

void run_real_block_fp8_tp2_slice_oracle(const char* artifact_path) {
    constexpr std::int32_t kAttentionRows = 14336;
    constexpr std::int32_t kAttentionColumns = 5120;
    constexpr std::array<ninfer::artifact::SliceRange, 4> kAttentionShard0 = {
        ninfer::artifact::SliceRange{0, 3072},
        ninfer::artifact::SliceRange{6144, 512},
        ninfer::artifact::SliceRange{7168, 3072},
        ninfer::artifact::SliceRange{13312, 512}};
    constexpr std::array<ninfer::artifact::SliceRange, 4> kAttentionShard1 = {
        ninfer::artifact::SliceRange{3072, 3072},
        ninfer::artifact::SliceRange{6656, 512},
        ninfer::artifact::SliceRange{10240, 3072},
        ninfer::artifact::SliceRange{13824, 512}};
    ninfer::artifact::Reader reader(artifact_path);
    const std::vector<std::byte> attention = real_block_payload(
        reader, "text/layers/3/attention/query_key_gate_value", kAttentionRows,
        kAttentionColumns);
    const std::array<std::array<ninfer::artifact::SliceRange, 4>, 2> attention_ranges = {
        kAttentionShard0, kAttentionShard1};
    const std::array<std::uint64_t, 2> attention_shape = {
        static_cast<std::uint64_t>(kAttentionRows), static_cast<std::uint64_t>(kAttentionColumns)};
    for (std::size_t device = 0; device < attention_ranges.size(); ++device) {
        const auto slice = ninfer::artifact::tensor_row_slice(
            ninfer::artifact::StorageLayout::BlockScaleM128K128V1,
            ninfer::artifact::NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
            attention_shape,
            attention_ranges[device], attention);
        const auto actual = apply_tensor_slice(attention, slice);
        const auto expected = expected_block_row_shard(attention, kAttentionRows,
                                                       kAttentionColumns, attention_ranges[device]);
        require(actual == expected, "real block-FP8 attention TP2 row shard differs from oracle");
    }

    constexpr std::int32_t kDownRows = 5120;
    constexpr std::int32_t kDownColumns = 17408;
    const std::vector<std::byte> down =
        real_block_payload(reader, "text/layers/0/mlp/down", kDownRows, kDownColumns);
    const std::array<std::uint64_t, 2> down_shape = {
        static_cast<std::uint64_t>(kDownRows), static_cast<std::uint64_t>(kDownColumns)};
    for (const ninfer::artifact::SliceRange range :
         std::array<ninfer::artifact::SliceRange, 2>{
             ninfer::artifact::SliceRange{0, 8704}, ninfer::artifact::SliceRange{8704, 8704}}) {
        const auto slice = ninfer::artifact::tensor_column_slice(
            ninfer::artifact::StorageLayout::BlockScaleM128K128V1,
            ninfer::artifact::NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
            down_shape,
            std::array<ninfer::artifact::SliceRange, 1>{range}, down);
        const auto actual = apply_tensor_slice(down, slice);
        const auto expected = expected_block_column_shard(down, kDownRows, kDownColumns, range);
        require(actual == expected, "real block-FP8 MLP down TP2 column shard differs from oracle");
    }
    std::cout << "real block-FP8 TP2 shard oracle: passed\n";
}

void run_real_block_fp8_gdn_oracle(const char* artifact_path, std::int32_t tokens) {
    constexpr std::int32_t kRows = 16384;
    constexpr std::int32_t kColumns = 5120;
    const std::int32_t kTokens = tokens;
    ninfer::artifact::Reader reader(artifact_path);
    const std::vector<std::byte> payload = real_block_payload(
        reader, "text/layers/0/gdn/query_key_value_z", kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * kTokens);
    std::vector<std::uint16_t> qkv(static_cast<std::size_t>(10240) * kTokens);
    std::vector<std::uint16_t> z(static_cast<std::size_t>(6144) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 23U) % 41U) - 20) / 256.0F;
        input[index] = float_to_bf16(value);
    }

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_qkv(qkv.size() * sizeof(std::uint16_t));
    DeviceAllocation device_z(z.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, payload.size(), kRows, kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, kTokens});
    ninfer::Tensor qkv_output(device_qkv.pointer, ninfer::DType::BF16, {10240, kTokens});
    ninfer::Tensor z_output(device_z.pointer, ninfer::DType::BF16, {6144, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::gdn_input_proj(x, weight, qkv_output, z_output,
                                ninfer::ops::LinearPolicy::A16Only, workspace, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(qkv.data(), device_qkv.pointer, qkv.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(z.data(), device_z.pointer, z.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));

    constexpr std::array<int, 6> kRowsToCheck = {0, 127, 128, 10239, 10240, 16383};
    double maximum_error = 0.0;
    for (const int token : std::array<int, 5>{
             0, tokens / 2, std::min(127, tokens - 1), std::min(128, tokens - 1), tokens - 1}) {
        for (const int row : kRowsToCheck) {
            const double expected = block_linear_fp64(payload, input, row, token, kRows, kColumns);
            const double actual = row < 10240
                                      ? bf16_to_double(qkv[static_cast<std::size_t>(token) * 10240 + row])
                                      : bf16_to_double(z[static_cast<std::size_t>(token) * 6144 + row - 10240]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    "real block-FP8 GDN projection differs from the FP64 oracle");
        }
    }
    std::cout << "real block-FP8 GDN oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_real_block_fp8_attention_oracle(const char* artifact_path, const char* object_name,
                                         const char* label, std::int32_t tokens) {
    constexpr std::int32_t kRows = 14336;
    constexpr std::int32_t kColumns = 5120;
    const std::int32_t kTokens = tokens;
    ninfer::artifact::Reader reader(artifact_path);
    const std::vector<std::byte> payload = real_block_payload(
        reader, object_name, kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * kTokens);
    std::array<std::vector<std::uint16_t>, 4> outputs{
        std::vector<std::uint16_t>(6144U * kTokens),
        std::vector<std::uint16_t>(6144U * kTokens),
        std::vector<std::uint16_t>(1024U * kTokens),
        std::vector<std::uint16_t>(1024U * kTokens)};
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 19U) % 37U) - 18) / 256.0F;
        input[index] = float_to_bf16(value);
    }

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    std::array<DeviceAllocation, 4> device_outputs{
        DeviceAllocation(outputs[0].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[1].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[2].size() * sizeof(std::uint16_t)),
        DeviceAllocation(outputs[3].size() * sizeof(std::uint16_t))};
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, payload.size(), kRows, kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, kTokens});
    ninfer::Tensor query(device_outputs[0].pointer, ninfer::DType::BF16, {6144, kTokens});
    ninfer::Tensor gate(device_outputs[1].pointer, ninfer::DType::BF16, {6144, kTokens});
    ninfer::Tensor key(device_outputs[2].pointer, ninfer::DType::BF16, {1024, kTokens});
    ninfer::Tensor value(device_outputs[3].pointer, ninfer::DType::BF16, {1024, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::attn_input_proj(x, weight, query, gate, key, value,
                                 ninfer::ops::LinearPolicy::A16Only, workspace, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t output = 0; output < outputs.size(); ++output) {
        CUDA_CHECK(cudaMemcpy(outputs[output].data(), device_outputs[output].pointer,
                              outputs[output].size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
    }

    constexpr std::array<int, 9> kRowsToCheck = {0, 127, 6143, 6144, 7167, 7168, 13311, 13312,
                                                  14335};
    double maximum_error = 0.0;
    for (const int token : std::array<int, 5>{
             0, tokens / 2, std::min(127, tokens - 1), std::min(128, tokens - 1), tokens - 1}) {
        for (const int row : kRowsToCheck) {
            std::size_t output_index = 0;
            int output_row = 0;
            int output_rows = 0;
            if (row < 6144) {
                output_index = 0;
                output_row = row;
                output_rows = 6144;
            } else if (row < 7168) {
                output_index = 2;
                output_row = row - 6144;
                output_rows = 1024;
            } else if (row < 13312) {
                output_index = 1;
                output_row = row - 7168;
                output_rows = 6144;
            } else {
                output_index = 3;
                output_row = row - 13312;
                output_rows = 1024;
            }
            const double expected = block_linear_fp64(payload, input, row, token, kRows, kColumns);
            const double actual = bf16_to_double(
                outputs[output_index][static_cast<std::size_t>(token) * output_rows + output_row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    "real block-FP8 attention projection differs from the FP64 oracle");
        }
    }
    std::cout << "real block-FP8 " << label << " oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_real_block_fp8_swiglu_oracle(const char* artifact_path, const char* object_name,
                                      const char* label, std::int32_t tokens) {
    constexpr std::int32_t kRows = 34816;
    constexpr std::int32_t kColumns = 5120;
    const std::int32_t kTokens = tokens;
    constexpr std::int32_t kIntermediateRows = 17408;
    ninfer::artifact::Reader reader(artifact_path);
    const std::vector<std::byte> payload =
        real_block_payload(reader, object_name, kRows, kColumns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kColumns) * kTokens);
    std::vector<std::uint16_t> output(static_cast<std::size_t>(kIntermediateRows) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 17U) % 31U) - 15) / 128.0F;
        input[index] = float_to_bf16(value);
    }

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_output(output.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, payload.size(), kRows, kColumns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {kColumns, kTokens});
    ninfer::Tensor result(device_output.pointer, ninfer::DType::BF16,
                          {kIntermediateRows, kTokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::linear_swiglu(x, weight, result, ninfer::ops::LinearPolicy::A16Only, workspace,
                               nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.pointer,
                          output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    constexpr std::array<int, 6> kRowsToCheck = {0, 127, 128, 8703, 8704, 17407};
    double maximum_error = 0.0;
    for (const int token : std::array<int, 5>{
             0, tokens / 2, std::min(127, tokens - 1), std::min(128, tokens - 1), tokens - 1}) {
        for (const int row : kRowsToCheck) {
            const double gate = block_linear_fp64(payload, input, row, token, kRows, kColumns);
            const double up =
                block_linear_fp64(payload, input, row, token, kRows, kColumns, kIntermediateRows);
            const double expected = gate / (1.0 + std::exp(-gate)) * up;
            const double actual =
                bf16_to_double(output[static_cast<std::size_t>(token) * kIntermediateRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                    "real block-FP8 SwiGLU differs from the FP64 oracle");
        }
    }
    std::cout << "real block-FP8 " << label << " oracle (tokens=" << tokens
              << "): passed; max_abs_error=" << maximum_error << '\n';
}

void run_real_block_fp8_projection_oracle(const char* artifact_path, const char* object_name,
                                         std::int32_t columns, const char* label,
                                         std::int32_t tokens, bool add_residual) {
    constexpr std::int32_t kRows = 5120;
    const std::int32_t kTokens = tokens;
    ninfer::artifact::Reader reader(artifact_path);
    const std::vector<std::byte> payload =
        real_block_payload(reader, object_name, kRows, columns);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(columns) * kTokens);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(kRows) * kTokens);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 11U) % 29U) - 14) / 256.0F;
        input[index] = float_to_bf16(value);
    }
    for (std::size_t index = 0; index < residual.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 7U) % 19U) - 9) / 128.0F;
        residual[index] = float_to_bf16(value);
    }
    const std::vector<std::uint16_t> initial_residual = residual;

    DeviceAllocation device_payload(payload.size());
    DeviceAllocation device_input(input.size() * sizeof(std::uint16_t));
    DeviceAllocation device_residual(residual.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device_payload.pointer, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_residual.pointer, residual.data(),
                          residual.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    const ninfer::Weight weight =
        make_block_weight(device_payload.pointer, payload.size(), kRows, columns);
    const ninfer::Tensor x(device_input.pointer, ninfer::DType::BF16, {columns, kTokens});
    ninfer::Tensor result(device_residual.pointer, ninfer::DType::BF16, {kRows, kTokens});
    ninfer::WorkspaceArena workspace(256);
    if (add_residual) {
        ninfer::ops::linear_add(x, weight, result, ninfer::ops::LinearPolicy::A16Only, workspace,
                                nullptr);
    } else {
        ninfer::ops::linear(x, weight, result, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(residual.data(), device_residual.pointer,
                          residual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    constexpr std::array<int, 4> kRowsToCheck = {0, 127, 128, 5119};
    double maximum_error = 0.0;
    for (const int token : std::array<int, 5>{
             0, tokens / 2, std::min(127, tokens - 1), std::min(128, tokens - 1), tokens - 1}) {
        for (const int row : kRowsToCheck) {
            const double expected =
                block_linear_fp64(payload, input, row, token, kRows, columns) +
                (add_residual
                     ? bf16_to_double(initial_residual[static_cast<std::size_t>(token) * kRows + row])
                     : 0.0);
            const double actual =
                bf16_to_double(residual[static_cast<std::size_t>(token) * kRows + row]);
            const double error = std::abs(actual - expected);
            maximum_error = std::max(maximum_error, error);
            require(error <= 0.005 + 0.004 * std::abs(expected),
                                        add_residual ? "real block-FP8 LinearAdd differs from the FP64 oracle"
                                                                 : "real block-FP8 Linear differs from the FP64 oracle");
        }
    }
        std::cout << "real block-FP8 " << label << " oracle (tokens=" << tokens
                            << "): passed; max_abs_error=" << maximum_error << '\n';
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
        (count_error == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        CUDA_CHECK(count_error);
        for (const std::int32_t tokens : {4, 5, 8, 9}) {
            run_block_fp8_linear_oracle(tokens);
        }
        run_marlin_block_fp8_linear_oracle(5);
        run_marlin_block_fp8_linear_oracle(12);
        run_marlin_block_fp8_swiglu_oracle(1);
        run_marlin_block_fp8_swiglu_oracle(4);
        run_marlin_block_fp8_swiglu_oracle(12);
        run_marlin_block_fp8_linear_add_oracle();
        run_marlin_block_fp8_attention_oracle();
        // The 128-token activation tile is staged independently of the 64-row weight
        // tile; a wider token block is what exercises the tail of that staging.
        run_block_fp8_linear_oracle(192);
        for (const std::int32_t tokens : {4, 5, 8, 9, 192}) {
            run_block_fp8_linear_broadcast_oracle(tokens);
        }
        for (const std::int32_t tokens : {4, 5, 8, 9, 192}) {
            run_block_fp8_linear_packed_oracle(tokens);
        }
        for (const std::int32_t tokens : {4, 5, 8, 9, 192}) {
            run_block_fp8_linear_packed256_oracle(tokens);
        }
        for (const std::int32_t tokens : {4, 5, 8, 9}) {
            run_block_fp8_linear_add_oracle(tokens);
            run_block_fp8_swiglu_oracle(tokens);
            run_block_fp8_attention_oracle(tokens);
        }
        run_block_fp8_swiglu_oracle(192);
        for (const std::int32_t tokens : {4, 5, 8, 9, 192}) {
            run_block_fp8_swiglu_packed_oracle(tokens);
        }
        run_block_fp8_attention_oracle(192);
        for (const std::int32_t tokens : {4, 5, 12, 13, 192}) {
            run_block_fp8_gdn_oracle(tokens);
        }
        run_block_fp8_linear_pair_oracle();
        if (const char* artifact_path = std::getenv("NINFER_FP8_BLOCK_ARTIFACT");
            artifact_path != nullptr && *artifact_path != '\0') {
            run_real_block_fp8_tp2_slice_oracle(artifact_path);
            for (const std::int32_t tokens : {1, 4, 5, 12, 13, 192}) {
                run_real_block_fp8_gdn_oracle(artifact_path, tokens);
            }
            for (const std::int32_t tokens : {1, 4, 5, 9, 192}) {
                run_real_block_fp8_attention_oracle(
                    artifact_path, "text/layers/3/attention/query_key_gate_value", "attention",
                    tokens);
                run_real_block_fp8_swiglu_oracle(artifact_path, "text/layers/0/mlp/gate_up",
                                                "SwiGLU", tokens);
                run_real_block_fp8_projection_oracle(artifact_path, "text/layers/0/mlp/down",
                                                     17408, "LinearAdd", tokens, true);
                run_real_block_fp8_attention_oracle(
                    artifact_path, "mtp/layer/attention/query_key_gate_value", "MTP attention",
                    tokens);
                run_real_block_fp8_swiglu_oracle(artifact_path, "mtp/layer/mlp/gate_up",
                                                "MTP SwiGLU", tokens);
                run_real_block_fp8_projection_oracle(artifact_path, "mtp/layer/attention/output",
                                                     6144, "MTP attention LinearAdd", tokens, true);
                run_real_block_fp8_projection_oracle(artifact_path, "mtp/layer/mlp/down", 17408,
                                                     "MTP MLP LinearAdd", tokens, true);
            }
            for (const std::int32_t tokens : {12, 13}) {
                run_real_block_fp8_projection_oracle(artifact_path, "text/layers/0/mlp/down",
                                                     17408, "MLP down Linear", tokens, false);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}