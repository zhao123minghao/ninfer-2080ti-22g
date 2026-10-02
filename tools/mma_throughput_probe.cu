// Standalone tensor-core throughput probe for sm_75 (Turing).
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_75 tools/mma_throughput_probe.cu -o mma_throughput_probe
//   CUDA_VISIBLE_DEVICES=0 ./mma_throughput_probe
//
// What this establishes
// ---------------------
// Turing's combined FP16/INT8 tensor core is reached through three PRECISION- AND
// ACCUMULATOR-SPECIFIC MMA forms, and they do not run at the same rate. NInfer's prefill GEMM uses
// `m16n8k8.f32.f16.f16.f32`; the grouped signed-integer weights this target registers happen to be
// representable exactly in int8, and `m8n8k16.s32.s8.s8.s32` is 3.97x faster. This probe measures
// that ratio on the actual card instead of quoting the 107.6 TFLOP/s marketing figure, which is the
// fp16-with-FP16-ACCUMULATE number and therefore 2x above what an FP32-accumulating kernel can
// ever reach.
//
// Absolute TFLOP/s is meaningless on this part without the clock it was taken at: an idle card sits
// at 300 MHz, a tensor-core-only loop boosts to ~1860 MHz, and a real prefill (which also moves
// DRAM traffic and is therefore power-limited) runs at 1665-1725 MHz. Every measurement here
// reports the effective SM clock sampled from `clock64()` in the same window, and the
// clock-invariant form -- FLOP per SM per cycle -- is the number to compare. The three pure-MMA
// rows land exactly on the TU102 architectural rates (256, 512 and 2048 ops/SM/cycle), which is
// what makes them a valid ceiling for a real kernel.
//
// The fourth row exists because the ratio alone is not the answer. A Q4G64/Q5G64 weight scale is
// indexed by output row, while the int8 B operand is shared across all output rows, so the scale
// cannot be folded into either operand: it has to be folded out of the int32 accumulator at every
// 64-wide K group boundary, which is the pattern `src/ops/kernel/gqa_attention_prefill_i8.cuh`
// already uses for its QK^T scores. That row measures the cost of the mandatory epilogue by
// mirroring its instruction mix against register-resident fragments.
//
//   See docs/maintainer/tensor-formats.md section 7.2 for the code/scale semantics, and
//   history.md section 9 for how these numbers are used.

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

namespace {

// Independent accumulator chains per warp. Enough in flight to cover the tensor core's latency
// without spilling.
constexpr int kChains = 16;

// MACs retired per warp instruction. All three forms are 1024: 16x8x8 for `m16n8k8`, 8x8x16 for
// `m8n8k16`.
constexpr int kMacs = 1024;

// The GEMM-shaped case: kGemmBlocks independent m8n8k16 output tiles, kGemmMmas of them per 64-wide
// K group, then the group epilogue. 4 x 16 = 64 is exactly one Q4G64/Q5G64 group.
constexpr int kGemmBlocks = 8;
constexpr int kGemmMmas   = 4;
constexpr int kGemmMacs   = kGemmBlocks * kGemmMmas * kMacs;

__global__ void k_f16_f32(unsigned long long* out, int iters) {
    unsigned a0 = 0x3c003c00u;
    unsigned a1 = 0x3c003c00u;
    unsigned b0 = 0x3c003c00u;
    float c[kChains][4] = {};
    __syncwarp();
    const unsigned long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kChains; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
                "{%0,%1,%2,%3};"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(a0), "r"(a1), "r"(b0));
        }
    }
    const unsigned long long t1 = clock64();
    float s = 0.0f;
    for (int j = 0; j < kChains; ++j) {
        for (int m = 0; m < 4; ++m) {
            s += c[j][m];
        }
    }
    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        if (s == 12345.678f) {
            out[1] = 1;
        }
    }
}

// Informational: identifies what the advertised 107.6 TFLOP/s figure is quoted against.
__global__ void k_f16_f16(unsigned long long* out, int iters) {
    unsigned a0 = 0x3c003c00u;
    unsigned a1 = 0x3c003c00u;
    unsigned b0 = 0x3c003c00u;
    unsigned c[kChains][2] = {};
    __syncwarp();
    const unsigned long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kChains; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, "
                "{%0,%1};"
                : "+r"(c[j][0]), "+r"(c[j][1])
                : "r"(a0), "r"(a1), "r"(b0));
        }
    }
    const unsigned long long t1 = clock64();
    unsigned s = 0;
    for (int j = 0; j < kChains; ++j) {
        s += c[j][0] + c[j][1];
    }
    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        if (s == 12345u) {
            out[1] = 1;
        }
    }
}

__global__ void k_s8_s32(unsigned long long* out, int iters) {
    unsigned a0 = 0x01010101u;
    unsigned b0 = 0x01010101u;
    int c[kChains][2] = {};
    __syncwarp();
    const unsigned long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < kChains; ++j) {
            asm volatile(
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                : "+r"(c[j][0]), "+r"(c[j][1])
                : "r"(a0), "r"(b0));
        }
    }
    const unsigned long long t1 = clock64();
    int s = 0;
    for (int j = 0; j < kChains; ++j) {
        s += c[j][0] + c[j][1];
    }
    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        if (s == 12345) {
            out[1] = 1;
        }
    }
}

__device__ __forceinline__ void mma_s8_m8n8k16(int& c0, int& c1, unsigned a0, unsigned b0) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
}

// The same instruction with a dedicated zero accumulator pair, so opening a group costs no separate
// clearing step. That is how the real kernel starts each group's int32 partial.
__device__ __forceinline__ void mma_s8_m8n8k16_zero(int& c0, int& c1, unsigned a0, unsigned b0,
                                                    unsigned z0, unsigned z1) {
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
        : "=r"(c0), "=r"(c1)
        : "r"(a0), "r"(b0), "r"(z0), "r"(z1));
}

__global__ void k_s8_gemm_shape(unsigned long long* out, int groups) {
    unsigned a[kGemmMmas];
    unsigned b[kGemmBlocks];
#pragma unroll
    for (int i = 0; i < kGemmMmas; ++i) {
        a[i] = 0x01010101u;
    }
#pragma unroll
    for (int i = 0; i < kGemmBlocks; ++i) {
        b[i] = 0x01010101u;
    }
    const unsigned zero = 0u;
    int acc[kGemmBlocks][2];
    float facc[kGemmBlocks][2] = {};
#pragma unroll
    for (int i = 0; i < kGemmBlocks; ++i) {
        acc[i][0] = 0;
        acc[i][1] = 0;
    }

    __syncwarp();
    const unsigned long long t0 = clock64();
    for (int g = 0; g < groups; ++g) {
#pragma unroll
        for (int t = 0; t < kGemmBlocks; ++t) {
            mma_s8_m8n8k16_zero(acc[t][0], acc[t][1], a[0], b[t], zero, zero);
#pragma unroll
            for (int k = 1; k < kGemmMmas; ++k) {
                mma_s8_m8n8k16(acc[t][0], acc[t][1], a[k], b[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < kGemmBlocks; ++t) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                facc[t][c] = __fmaf_rn(static_cast<float>(acc[t][c]), 1.0f, facc[t][c]);
            }
        }
    }
    const unsigned long long t1 = clock64();

    float s = 0.0f;
#pragma unroll
    for (int t = 0; t < kGemmBlocks; ++t) {
        s += facc[t][0] + facc[t][1] + static_cast<float>(acc[t][0] + acc[t][1]);
    }
    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
        if (s == 12345.678f) {
            out[1] = 1;
        }
    }
}

struct Case {
    const char* name;
    void (*kernel)(unsigned long long*, int);
    double macs_per_warp_iter;
};

struct Measurement {
    double tflops;
    double clock_ghz;
};

// The best of `repeats` runs, together with the SM clock sampled inside that same run. clock64()
// counts SM cycles and every SM shares one clock domain, so a single block's reading is a valid
// estimate; taking the best run keeps clock ramping out of the reported figure.
Measurement measure(const Case& c, int blocks, int threads, int iters, int repeats) {
    unsigned long long* dev = nullptr;
    CK(cudaMalloc(&dev, 2 * sizeof(unsigned long long)));
    for (int w = 0; w < 2; ++w) { c.kernel<<<blocks, threads>>>(dev, iters); }
    CK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));

    Measurement best{0.0, 0.0};
    for (int r = 0; r < repeats; ++r) {
        CK(cudaMemset(dev, 0, 2 * sizeof(unsigned long long)));
        CK(cudaEventRecord(start));
        c.kernel<<<blocks, threads>>>(dev, iters);
        CK(cudaEventRecord(stop));
        CK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, start, stop));
        unsigned long long cycles = 0;
        CK(cudaMemcpy(&cycles, dev, sizeof(cycles), cudaMemcpyDeviceToHost));

        const double seconds = ms * 1e-3;
        const double warps   = static_cast<double>(blocks) * (threads / 32.0);
        const double tflops =
            warps * static_cast<double>(iters) * c.macs_per_warp_iter * 2.0 / seconds / 1e12;
        if (tflops > best.tflops) {
            best = {tflops, static_cast<double>(cycles) / seconds / 1e9};
        }
    }
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    CK(cudaFree(dev));
    return best;
}

} // namespace

int main() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device : %s  SMs=%d  max boost %d MHz\n", prop.name, prop.multiProcessorCount,
                prop.clockRate / 1000);
    const int sms     = prop.multiProcessorCount;
    const int threads = 256;
    const int blocks  = sms * 3;
    const int iters   = 20000;
    const int repeats = 5;

    const Case cases[] = {
        {"m16n8k8.f32.f16.f16.f32", k_f16_f32, static_cast<double>(kChains) * kMacs},
        {"m16n8k8.f16.f16.f16.f16", k_f16_f16, static_cast<double>(kChains) * kMacs},
        {"m8n8k16.s32.s8.s8.s32  ", k_s8_s32, static_cast<double>(kChains) * kMacs},
        {"  + 64K-group epilogue ", k_s8_gemm_shape, static_cast<double>(kGemmMacs)},
    };

    // The ratio is taken from the clock-invariant column, not from TFLOP/s: the first case of a
    // run often executes before the card has finished ramping, and comparing its TFLOP/s against a
    // later case's would understate every subsequent ratio.
    double baseline = 0.0;
    for (const Case& c : cases) {
        const Measurement m       = measure(c, blocks, threads, iters, repeats);
        const double per_sm_cycle = m.tflops * 1e12 / (m.clock_ghz * 1e9) / sms;
        std::printf("%s : %8.2f TFLOP/s @ %4.0f MHz = %6.0f FLOP/SM/cycle", c.name, m.tflops,
                    m.clock_ghz * 1000.0, per_sm_cycle);
        if (baseline == 0.0) {
            baseline = per_sm_cycle;
            std::printf("\n");
        } else {
            std::printf("   %+.2fx\n", per_sm_cycle / baseline);
        }
    }
    return 0;
}
