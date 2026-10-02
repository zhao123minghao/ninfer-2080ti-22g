#include "ops/common/mma.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace ninfer::test {
namespace {

constexpr int kRows = 16;
constexpr int kCols = 8;
constexpr int kDepth = 256;
constexpr int kGroupDepth = 64;
constexpr int kChunksPerGroup = kGroupDepth / 32;
constexpr int kGroups = kDepth / kGroupDepth;
constexpr int kWordsPerRow = kDepth / 2;

__global__ void mma_s8_prefill_fragment_kernel(const std::int8_t* input_a,
                                              const std::int8_t* input_b,
                                              std::int32_t* output) {
    __shared__ __align__(16) std::int8_t a_smem[kRows * kWordsPerRow * 2];
    __shared__ __align__(16) std::int8_t b_smem[kCols * kWordsPerRow * 2];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    for (int index = lane; index < kRows * kDepth; index += 32) {
        const int row = index / kDepth;
        const int col = index - row * kDepth;
        const int word = ops::gqa_prefill_swz(row, col >> 1);
        a_smem[(row * kWordsPerRow + word) * 2 + (col & 1)] = input_a[index];
    }
    for (int index = lane; index < kCols * kDepth; index += 32) {
        const int col = index / kDepth;
        const int depth = index - col * kDepth;
        const int word = ops::gqa_prefill_swz(col, depth >> 1);
        b_smem[(col * kWordsPerRow + word) * 2 + (depth & 1)] =
            input_b[depth * kCols + col];
    }
    __syncthreads();

    int a_row, a_col;
    ops::gqa_prefill_i8_sm75_a_fragment_base(lane, a_row, a_col);
    const int b_row = lane & 7;
    const int b_col = ((lane >> 3) & 1) << 3;

    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int group = 0; group < kGroups; ++group) {
        for (int chunk = 0; chunk < kChunksPerGroup; ++chunk) {
            const int k = group * kChunksPerGroup + chunk;
            const int acol = k * 16 + a_col;
            const int bcol = k * 16 + b_col;
            unsigned a0, a1, a2, a3;
            ops::ldmatrix_x4(
                a0, a1, a2, a3,
                ops::smem_addr(&a_smem[(a_row * kWordsPerRow +
                                        ops::gqa_prefill_swz(a_row, acol)) *
                                       2]));
            unsigned b0, b1;
            ops::ldmatrix_x2(
                b0, b1,
                ops::smem_addr(&b_smem[(b_row * kWordsPerRow +
                                        ops::gqa_prefill_swz(b_row, bcol)) *
                                       2]));
            ops::mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
        }
    }

    const int row = lane >> 2;
    const int col = (lane & 3) << 1;
    output[row * kCols + col] = c0;
    output[row * kCols + col + 1] = c1;
    output[(row + 8) * kCols + col] = c2;
    output[(row + 8) * kCols + col + 1] = c3;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

} // namespace
} // namespace ninfer::test

int main() {
#if !defined(NINFER_SM75)
    std::cout << "SKIP: mma_s8 Turing decomposition is only compiled for sm_75\n";
    return 77;
#else
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    using namespace ninfer::test;
    std::int8_t host_a[kRows * kDepth];
    std::int8_t host_b[kDepth * kCols];
    std::int32_t reference[kRows * kCols]{};
    for (int index = 0; index < kRows * kDepth; ++index) {
        host_a[index] = static_cast<std::int8_t>((index * 7 + 3) % 31 - 15);
    }
    for (int index = 0; index < kDepth * kCols; ++index) {
        host_b[index] = static_cast<std::int8_t>((index * 11 + 5) % 29 - 14);
    }
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            std::int32_t sum = 0;
            for (int depth = 0; depth < kDepth; ++depth) {
                sum += static_cast<std::int32_t>(host_a[row * kDepth + depth]) *
                       static_cast<std::int32_t>(host_b[depth * kCols + col]);
            }
            reference[row * kCols + col] = sum;
        }
    }

    std::int8_t* device_a = nullptr;
    std::int8_t* device_b = nullptr;
    std::int32_t* device_output = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), sizeof(host_a)), "cudaMalloc A");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), sizeof(host_b)), "cudaMalloc B");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(reference)),
               "cudaMalloc output");
    check_cuda(cudaMemcpy(device_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice), "copy A");
    check_cuda(cudaMemcpy(device_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice), "copy B");

    mma_s8_prefill_fragment_kernel<<<1, 32>>>(device_a, device_b, device_output);
    check_cuda(cudaGetLastError(), "launch mma_s8 test");
    check_cuda(cudaDeviceSynchronize(), "synchronize mma_s8 test");

    std::int32_t actual[kRows * kCols];
    check_cuda(cudaMemcpy(actual, device_output, sizeof(actual), cudaMemcpyDeviceToHost),
               "copy output");
    int mismatches = 0;
    int max_abs_diff = 0;
    for (int index = 0; index < kRows * kCols; ++index) {
        const int diff = std::abs(actual[index] - reference[index]);
        mismatches += diff != 0;
        max_abs_diff = std::max(max_abs_diff, diff);
    }

    check_cuda(cudaFree(device_a), "free A");
    check_cuda(cudaFree(device_b), "free B");
    check_cuda(cudaFree(device_output), "free output");
    if (mismatches != 0) {
        std::cerr << "mma_s8 sm_75 fragment oracle: mismatches=" << mismatches << "/"
                  << kRows * kCols << " max_abs_diff=" << max_abs_diff << '\n';
        return 1;
    }
    std::cout << "mma_s8 sm_75 fragment oracle exact for M=16 N=8 K=256\n";
    return 0;
#endif
}