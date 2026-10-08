// Which ldmatrix address pattern produces the A register order mma_s8 expects?
//
// Build: nvcc -O3 -std=c++17 -arch=sm_75 .scratch/frag_probe.cu -o /tmp/frag_probe && /tmp/frag_probe
//
// src/ops/common/mma.cuh::mma_s8 emulates m16n8k32 on Turing as four m8n8k16 sub-MMAs, which
// requires a0 = (rows 0-7, k 0-15), a1 = (rows 0-7, k 16-31), a2 = (rows 8-15, k 0-15),
// a3 = (rows 8-15, k 16-31). gqa_attention_prefill_i8.cuh addresses its A fragments with
// row = (lane&7) + ((mat&1)<<3), byte = (mat>>1)*16, which reads as (M0K0, M1K0, M0K1, M1K1) --
// the transposed assignment. Exactly one of the two can be right; this settles it against a naive
// int32 reference rather than by reading the PTX tables.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(expr)                                                                                   \
    do {                                                                                           \
        const cudaError_t error__ = (expr);                                                        \
        if (error__ != cudaSuccess) {                                                              \
            std::printf("CUDA error %s at line %d\n", cudaGetErrorString(error__), __LINE__);      \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_s8_m8n8k16(int& c0, int& c1, unsigned a0, unsigned b0) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
}

// Verbatim the sm_75 branch of mma.cuh::mma_s8.
__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    mma_s8_m8n8k16(c0, c1, a0, b0);
    mma_s8_m8n8k16(c0, c1, a1, b1);
    mma_s8_m8n8k16(c2, c3, a2, b0);
    mma_s8_m8n8k16(c2, c3, a3, b1);
}

constexpr int kRows = 16;
constexpr int kK    = 32;
constexpr int kCols = 8;

__global__ void frag_kernel(const std::int8_t* A, const std::int8_t* B, int* C, int pattern) {
    // A is [16][32] row-major int8, B is [8][32] row-major int8 (the .col B operand). Both rows are
    // padded to 32 bytes so every ldmatrix address is 16-byte aligned.
    __shared__ __align__(16) std::int8_t As[kRows * kK];
    __shared__ __align__(16) std::int8_t Bs[kCols * kK];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    for (int i = lane; i < kRows * kK; i += 32) { As[i] = A[i]; }
    for (int i = lane; i < kCols * kK; i += 32) { Bs[i] = B[i]; }
    __syncwarp();

    const int mat    = lane >> 3;
    const int row_in = lane & 7;
    int arow;
    int abyte;
    if (pattern == 0) { // attention-kernel assignment
        arow  = row_in + ((mat & 1) << 3);
        abyte = (mat >> 1) << 4;
    } else {            // transposed assignment
        arow  = row_in + ((mat >> 1) << 3);
        abyte = (mat & 1) << 4;
    }

    unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
    ldmatrix_x4(a0, a1, a2, a3, smem_addr(&As[arow * kK + abyte]));
    ldmatrix_x2(b0, b1, smem_addr(&Bs[(lane & 7) * kK + (((lane >> 3) & 1) << 4)]));

    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

    const int row = lane >> 2;
    const int col = 2 * (lane & 3);
    C[row * kCols + col]         = c0;
    C[row * kCols + col + 1]     = c1;
    C[(row + 8) * kCols + col]   = c2;
    C[(row + 8) * kCols + col + 1] = c3;
}

int main() {
    std::int8_t hA[kRows * kK];
    std::int8_t hB[kCols * kK];
    for (int i = 0; i < kRows * kK; ++i) { hA[i] = static_cast<std::int8_t>((i * 37 % 251) - 125); }
    for (int i = 0; i < kCols * kK; ++i) { hB[i] = static_cast<std::int8_t>((i * 53 % 241) - 120); }

    int ref[kRows * kCols] = {};
    for (int m = 0; m < kRows; ++m) {
        for (int n = 0; n < kCols; ++n) {
            std::int32_t acc = 0;
            for (int k = 0; k < kK; ++k) {
                acc += static_cast<std::int32_t>(hA[m * kK + k]) *
                       static_cast<std::int32_t>(hB[n * kK + k]);
            }
            ref[m * kCols + n] = acc;
        }
    }

    std::int8_t* dA   = nullptr;
    std::int8_t* dB   = nullptr;
    int* dC           = nullptr;
    CK(cudaMalloc(&dA, sizeof(hA)));
    CK(cudaMalloc(&dB, sizeof(hB)));
    CK(cudaMalloc(&dC, kRows * kCols * sizeof(int)));
    CK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));

    for (int pattern = 0; pattern < 2; ++pattern) {
        int hC[kRows * kCols] = {};
        CK(cudaMemset(dC, 0, kRows * kCols * sizeof(int)));
        frag_kernel<<<1, 32>>>(dA, dB, dC, pattern);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));
        long long worst = 0;
        for (int i = 0; i < kRows * kCols; ++i) {
            const long long d = static_cast<long long>(hC[i]) - ref[i];
            if (d > worst || -d > worst) { worst = d > 0 ? d : -d; }
        }
        std::printf("pattern %d (%s): max |diff| = %lld  %s\n", pattern,
                    pattern == 0 ? "attention-kernel assignment" : "transposed assignment", worst,
                    worst == 0 ? "<-- matches" : "");
    }
    return 0;
}
