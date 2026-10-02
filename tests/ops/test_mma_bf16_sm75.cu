//
// Independent oracle for the Turing (sm_75) m16n8k16 routes in ops/common/mma.cuh.
//
// Two production paths are checked directly against one naive FP64 oracle that evaluates the
// logical formula from the represented inputs:
//
//   * mma_bf16  — on sm_75 this restages its bf16 fragments into fp16 and runs the native fp16
//                 tensor core twice (see mma.cuh). The operand values here are already bf16, so
//                 the oracle multiplies the exact same numbers; the criterion is the accumulation
//                 difference between the tensor core's fp32 accumulate and the FP64 reference.
//   * mma_f16   — the m16n8k16 -> 2x m16n8k8 K-axis decomposition. Its two halves must be paired
//                 (a0,a1,b0) + (a2,a3,b1); the crossed pairing silently corrupts every k >= 8 term,
//                 which the wide-span and structured cases below detect as an O(1) relative error.
//
// The fragment geometry is built here from the PTX-standard m16n8k16 layout rather than copied
// from any production kernel, so a wrong assumption inside mma.cuh cannot mask itself.
//
// Scope of the magnitude cases: both routes require the operands to be exactly representable in
// fp16. bf16 -> fp16 is lossless across fp16's whole *normal* range (6.1035e-5 .. 65504), so the
// cases span and probe both ends of that range. Magnitudes below 6.1035e-5 are fp16-subnormal and
// would lose significand bits, so they are deliberately outside this test's contract.

#include "ops/common/mma.cuh"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace ninfer::test {
namespace {

constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 16;

// PTX m16n8k16 fragment geometry, with r = lane >> 2 and t = lane & 3. Identical for .bf16 and
// .f16 of the same shape.
//   A (16x16): a0={(r,2t),(r,2t+1)}  a1={(r+8,2t),(r+8,2t+1)}
//              a2={(r,8+2t),(r,9+2t)} a3={(r+8,8+2t),(r+9+2t)}
//   B (16x8):  b0={(2t,r),(2t+1,r)}   b1={(8+2t,r),(9+2t,r)}
//   C (16x8):  c0=(r,2t) c1=(r,2t+1) c2=(r+8,2t) c3=(r+8,2t+1)
__device__ __forceinline__ std::uint32_t pack_bf16_pair(float low, float high) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
    std::uint32_t packed;
    std::memcpy(&packed, &pair, sizeof(packed));
    return packed;
}

__device__ __forceinline__ std::uint32_t pack_f16_pair(float low, float high) {
    const __half2 pair = __floats2half2_rn(low, high);
    std::uint32_t packed;
    std::memcpy(&packed, &pair, sizeof(packed));
    return packed;
}

enum class Route { Bf16Restage, F16Direct };

// Loads the standard A/B fragments, runs the requested production route, and scatters C back to a
// dense [16,8] tile. One warp, one tile.
template <Route route>
__global__ void mma_tile_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int r    = lane >> 2;
    const int t    = lane & 3;

    const auto pack = [](float low, float high) {
        return route == Route::Bf16Restage ? pack_bf16_pair(low, high)
                                           : pack_f16_pair(low, high);
    };

    const std::uint32_t a0 = pack(a[r * kK + 2 * t], a[r * kK + 2 * t + 1]);
    const std::uint32_t a1 = pack(a[(r + 8) * kK + 2 * t], a[(r + 8) * kK + 2 * t + 1]);
    const std::uint32_t a2 = pack(a[r * kK + 8 + 2 * t], a[r * kK + 9 + 2 * t]);
    const std::uint32_t a3 = pack(a[(r + 8) * kK + 8 + 2 * t], a[(r + 8) * kK + 9 + 2 * t]);

    const std::uint32_t b0 = pack(b[(2 * t) * kN + r], b[(2 * t + 1) * kN + r]);
    const std::uint32_t b1 = pack(b[(8 + 2 * t) * kN + r], b[(9 + 2 * t) * kN + r]);

    float c0 = 0.0F;
    float c1 = 0.0F;
    float c2 = 0.0F;
    float c3 = 0.0F;
    if constexpr (route == Route::Bf16Restage) {
        ops::mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    } else {
        ops::mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }

    c[r * kN + 2 * t]           = c0;
    c[r * kN + 2 * t + 1]       = c1;
    c[(r + 8) * kN + 2 * t]     = c2;
    c[(r + 8) * kN + 2 * t + 1] = c3;
}

// C[m][n] = sum_k A[m][k] * B[k][n], evaluated in double from the represented inputs.
std::vector<double> tile_oracle(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<double> reference(static_cast<std::size_t>(kM) * kN, 0.0);
    for (int m = 0; m < kM; ++m) {
        for (int n = 0; n < kN; ++n) {
            double sum = 0.0;
            for (int k = 0; k < kK; ++k) {
                sum += static_cast<double>(a[static_cast<std::size_t>(m) * kK + k]) *
                       static_cast<double>(b[static_cast<std::size_t>(k) * kN + n]);
            }
            reference[static_cast<std::size_t>(m) * kN + n] = sum;
        }
    }
    return reference;
}

// Log-uniform magnitudes with an explicit zero fraction: real weights and activations span orders
// of magnitude, and exact zeros are the boundary the fragment geometry is most sensitive to.
void fill_log_uniform(std::vector<float>& values, std::uint32_t seed, double low, double high,
                      double zero_fraction) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> draw(0.0, 1.0);
    for (float& value : values) {
        if (draw(generator) < zero_fraction) {
            value = 0.0F;
            continue;
        }
        const double magnitude =
            std::exp(std::log(low) + draw(generator) * (std::log(high) - std::log(low)));
        value = static_cast<float>(draw(generator) < 0.5 ? -magnitude : magnitude);
    }
}

// Relative-L2 is the binding criterion: products are exact for both routes, so the only error
// source is fp32 accumulation versus the FP64 oracle (~1e-7 for 16 terms). The gross limiter is
// set at fp16-representable precision so it only catches genuinely wrong elements, including the
// O(1) corruption a crossed K pairing would produce.
constexpr ReductionCriterion kRouteCriterion{/*relative_l2*/ 1.0e-6, /*gross_absolute*/ 1.0e-12,
                                             /*gross_relative_to_max_reference*/ 1.0e-5};

int run_tile(std::string_view label, Route route, const std::vector<float>& a,
             const std::vector<float>& b) {
    const std::vector<double> reference = tile_oracle(a, b);

    const DeviceBuffer device_a = to_device_f32(a);
    const DeviceBuffer device_b = to_device_f32(b);
    const DeviceBuffer device_out(static_cast<std::size_t>(kM) * kN * sizeof(float));

    if (route == Route::Bf16Restage) {
        mma_tile_kernel<Route::Bf16Restage><<<1, 32>>>(static_cast<const float*>(device_a.p),
                                                       static_cast<const float*>(device_b.p),
                                                       static_cast<float*>(device_out.p));
    } else {
        mma_tile_kernel<Route::F16Direct><<<1, 32>>>(static_cast<const float*>(device_a.p),
                                                     static_cast<const float*>(device_b.p),
                                                     static_cast<float*>(device_out.p));
    }
    cuda_check_last_launch("mma_tile_kernel");
    cuda_synchronize();

    const std::vector<double> got =
        from_device_f32(device_out, static_cast<std::size_t>(kM) * kN);
    return verify_reduction(label, got, reference, kRouteCriterion);
}

int run_case(std::string_view label, Route route, double a_low, double a_high, double b_low,
             double b_high, std::uint32_t seed) {
    std::vector<float> a(static_cast<std::size_t>(kM) * kK);
    std::vector<float> b(static_cast<std::size_t>(kK) * kN);
    fill_log_uniform(a, seed, a_low, a_high, 0.1);
    fill_log_uniform(b, seed ^ 0x9e3779b9U, b_low, b_high, 0.1);
    // Both routes read bf16/fp16 operands directly; rounding to bf16 makes the inputs exactly
    // representable for both, so the oracle sees precisely the numbers the tensor core consumes.
    round_to_bf16(a);
    round_to_bf16(b);
    return run_tile(label, route, a, b);
}

} // namespace
} // namespace ninfer::test

int main() {
    using namespace ninfer::test;

#if !defined(NINFER_SM75)
    std::cout << "SKIP: mma.cuh's Turing route is only compiled when NINFER_SM75 is defined\n";
    return 77;
#else
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Realistic operand magnitudes: dequantized groupwise-int weights (|w| <= 1.6 in this
    // target's artifact) against activations spanning several orders of magnitude.
    failures += run_case("bf16 restage: weights vs activations", Route::Bf16Restage, 2.0e-4, 1.0e3,
                         2.0e-4, 1.6, 1);
    failures += run_case("f16 direct: weights vs activations", Route::F16Direct, 2.0e-4, 1.0e3,
                         2.0e-4, 1.6, 2);

    // Top of fp16's normal range: the restaging conversion must not saturate.
    failures += run_case("bf16 restage: near fp16 ceiling", Route::Bf16Restage, 1.0e-2, 5.0e4,
                         1.0e-2, 1.0, 3);
    failures += run_case("f16 direct: near fp16 ceiling", Route::F16Direct, 1.0e-2, 5.0e4, 1.0e-2,
                         1.0, 4);

    // Bottom of fp16's normal range: still lossless, so the criterion stays exact.
    failures += run_case("bf16 restage: near fp16 floor", Route::Bf16Restage, 1.0e-4, 1.0e-3,
                         1.0e-4, 1.0e-3, 5);

    // Structured: distinct k < 8 and k >= 8 halves isolate a mispaired K-axis decomposition.
    {
        std::vector<float> a(static_cast<std::size_t>(kM) * kK);
        std::vector<float> b(static_cast<std::size_t>(kK) * kN);
        for (int m = 0; m < kM; ++m) {
            for (int k = 0; k < kK; ++k) {
                a[static_cast<std::size_t>(m) * kK + k] =
                    static_cast<float>((k < 8 ? 1 : -1) * ((m % 5) + 1));
            }
        }
        for (int k = 0; k < kK; ++k) {
            for (int n = 0; n < kN; ++n) {
                b[static_cast<std::size_t>(k) * kN + n] =
                    static_cast<float>(k < 8 ? (n + 1) : 1.6F * (n + 2));
            }
        }
        round_to_bf16(a);
        round_to_bf16(b);
        failures += run_tile("bf16 restage: split K halves", Route::Bf16Restage, a, b);
    }

    if (failures == 0) {
        std::cout << "mma sm_75 routes match the FP64 oracle\n";
    }
    return failures == 0 ? 0 : 1;
#endif
}
