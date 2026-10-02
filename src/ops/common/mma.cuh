#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Every one of these wraps a raw PTX instruction that is Ampere+/Hopper+ only
// (ldmatrix: sm_75+; the various mma.sync shapes: sm_80+, mma_nvfp4_e4m3: Blackwell).
// This is deliberately the *only* file that inlines this PTX (see this port's
// full-tree audit) — every caller goes through these helpers rather than embedding its
// own asm, so this is the single boundary that needs a Volta guard for every kernel
// that's still tensor-core-only below sm_80 and doesn't yet have (or need, being wide-T/
// batching-shaped and unreachable in the decode-only MVP) its own SIMT replacement.
// Below sm_80 each helper traps instead of emitting PTX ptxas would reject outright:
// this is a compile-time necessity (invalid PTX fails ptxas regardless of whether the
// call is ever reached at runtime), not a "make it silently do nothing" shortcut — a
// kernel that's actually still on some reachable path here fails loudly at the trap,
// not silently with wrong output.

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#else
    r0 = 0;
    r1 = 0;
    __trap();
#endif
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#else
    r0 = 0;
    r1 = 0;
    r2 = 0;
    r3 = 0;
    __trap();
#endif
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#else
    r0 = 0;
    r1 = 0;
    __trap();
#endif
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#else
    r0 = 0;
    r1 = 0;
    r2 = 0;
    r3 = 0;
    __trap();
#endif
}

// Convert a 32-bit register holding 2x BF16 values to 2x FP16 values.
__device__ __forceinline__ unsigned bf162_to_f162(unsigned b) {
    const auto* bf_ptr = reinterpret_cast<const __nv_bfloat162*>(&b);
    float2 f = __bfloat1622float2(*bf_ptr);
    half2 h  = __float22half2_rn(f);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Scalar form of the widening above, for the element-wise staging paths.
__device__ __forceinline__ half bf16_to_f16(__nv_bfloat16 v) {
    return __float2half_rn(__bfloat162float(v));
}

// Convert one 16-byte KV vector (8x BF16) to the 8x FP16 values the KV cache stores.
//
// Every K/V write goes through this, because the cache is fp16 while the K/V activations are
// bf16. Doing it here rather than at every read is what makes the cache's element type the
// tensor-core operand type: the attention kernels stage cache vectors into shared memory and
// feed them straight to the mma, where the same bf16 -> fp16 widening would otherwise run once
// per staged element per key-tile walk. The widening itself is exact across fp16's normal range
// (bf16 carries 8 significand bits, fp16 carries 11), so the operands are bit-identical either
// way -- only the number of times the conversion executes changes.
__device__ __forceinline__ int4 bf16x8_to_f16x8(int4 raw) {
    int4 out;
    out.x = static_cast<int>(bf162_to_f162(static_cast<unsigned>(raw.x)));
    out.y = static_cast<int>(bf162_to_f162(static_cast<unsigned>(raw.y)));
    out.z = static_cast<int>(bf162_to_f162(static_cast<unsigned>(raw.z)));
    out.w = static_cast<int>(bf162_to_f162(static_cast<unsigned>(raw.w)));
    return out;
}

// Turing-native FP16 MMA: 16x8x8
__device__ __forceinline__ void mma_f16_m16n8k8(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Turing-native INT8 MMA: 8x8x16
__device__ __forceinline__ void mma_s8_m8n8k16(int& c0, int& c1, unsigned a0, unsigned b0) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
}

// Defined below; mma_bf16's Turing route restages its bf16 fragments into fp16 and runs this.
__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1);

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
// Turing has no bf16 tensor core, but it does have fp16 ones, and both operands here are bf16
// values already sitting in registers. A bf16 -> fp16 conversion is lossless across fp16's whole
// normal range (bf16 carries 8 significand bits, fp16 carries 11), so restaging the fragments and
// running the native m16n8k8 fp16 tensor core twice produces exactly the same products as
// decoding those bf16 values into fp32 did; only the accumulation association changes. The
// m16n8k16 A/B fragment layouts are identical for .bf16 and .f16, so this is a pure register
// restaging with no shuffles: ~20 issued instructions per tile against the emulation's ~160.
//
// Range: this target's grouped-int weights dequantize to |w| <= 1.6 over all 439 quantized
// tensors, and fp16's normal range reaches 65504, so no operand can saturate. The route is
// checked against a naive FP64 oracle in tests/ops/test_mma_bf16_sm75.cpp.
#elif defined(NINFER_SM75)
    mma_f16(c0, c1, c2, c3, bf162_to_f162(a0), bf162_to_f162(a1), bf162_to_f162(a2),
            bf162_to_f162(a3), bf162_to_f162(b0), bf162_to_f162(b1));
#else
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    __trap();
#endif
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
// m16n8k16 f16 is sm_80+; Turing has m16n8k8, so decompose into two.
//
// The seam is the K axis, and the two halves must be paired with their own fragments: for a
// PTX-standard m16n8k16 A fragment a0/a1 hold the k0..k7 half at rows r/r+8 and a2/a3 hold the
// k8..k15 half, while an m16n8k8 A fragment's two registers are exactly (rows 0..7, k0..7) and
// (rows 8..15, k0..7). So the k0..7 product takes (a0, a1) with the b0 half of B and the k8..15
// product takes (a2, a3) with b1. Verified with a standalone fragment probe against a naive
// per-element reference on sm_75: the pairing below reproduces the reference exactly
// (max |diff| 0) whereas the crossed (a0, a2)/(a1, a3) pairing differs by O(value).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(NINFER_SM75)
    mma_f16_m16n8k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_m16n8k8(c0, c1, c2, c3, a2, a3, b1);
#else
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    __trap();
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
// Turing has no m16n8k32 s8 MMA; decompose into four m8n8k16 ops.
#elif defined(NINFER_SM75)
    // Turing decomposition of m16n8k32 into 4x m8n8k16 operations:
    // (M0, K0), (M0, K1), (M1, K0), (M1, K1)
    mma_s8_m8n8k16(c0, c1, a0, b0);
    mma_s8_m8n8k16(c0, c1, a1, b1);
    mma_s8_m8n8k16(c2, c3, a2, b0);
    mma_s8_m8n8k16(c2, c3, a3, b1);
#else
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    __trap();
#endif
}

// kind::f8f6f4 requires sm_89; ptxas rejects the PTX outright below that, so the guard is
// needed at compile time and not merely at run time. This is the A8 path -- FP8 *activations* --
// which Volta has no hardware for at any width; the FP8 weight routes it accompanies are all
// A16 and need nothing from here. Same standing as mma_nvfp4_e4m3 below.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    __trap();
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
// Turing has no tf32 tensor core: repack the fragment onto the fp16 one, which it does have.
#elif defined(NINFER_SM75)
    // Both formats carry a 10-bit mantissa, so routing the tf32 fragment through the fp16 m16n8k8
    // MMA keeps the operand precision the Ampere/Ada builds get from mma.tf32, while replacing
    // 32 SHFL + 16 FFMA with 6 SHFL + 1 HMMA. Turing's operand magnitudes were measured on the
    // shipped 27B model (history.md 18.3): max|A| = 1.0 and max|B| = 55, i.e. about 1190x inside
    // fp16's normal range, so overflow is not reachable. C's element map is identical in both
    // fragment forms; the repack below is verified against the previous emulation by
    // .scratch/tf32_f16/repack_probe.cu (worst |diff| 1.8e-3 on |C| ~ 7 = fp16 operand rounding).
    //
    // Thread T (r = T / 4 in [0..7], t = T % 4 in [0..3]):
    // C: c0 = C(r, 2t), c1 = C(r, 2t+1), c2 = C(r+8, 2t), c3 = C(r+8, 2t+1)
    // tf32 A: a0 = A(r, t), a1 = A(r+8, t), a2 = A(r, t+4), a3 = A(r+8, t+4)
    // tf32 B: lane (tb = T / 8, j = T % 8) holds B(j, 2tb), B(j+4, 2tb) for j < 4 and
    //         B(j-4, 2tb+1), B(j, 2tb+1) for j >= 4
    // fp16 A: vec0 = {A(r, 2t), A(r, 2t+1)}, vec1 = {A(r+8, 2t), A(r+8, 2t+1)}
    // fp16 B: vec0 = {B(2t, r), B(2t+1, r)}
    const int lane    = threadIdx.x & 31;
    const int r       = lane >> 2;
    const int t       = lane & 3;

    const float fa0 = __uint_as_float(a0);
    const float fa1 = __uint_as_float(a1);
    const float fa2 = __uint_as_float(a2);
    const float fa3 = __uint_as_float(a3);
    const float fb0 = __uint_as_float(b0);
    const float fb1 = __uint_as_float(b1);

    const int a_src  = 4 * r + 2 * (t & 1);
    const bool a_hi  = (t >= 2);
    const __half2 a_pack0 = __floats2half2_rn(fa0, fa2);
    const __half2 a_pack1 = __floats2half2_rn(fa1, fa3);
    const unsigned a_pack0_bits = *reinterpret_cast<const unsigned*>(&a_pack0);
    const unsigned a_pack1_bits = *reinterpret_cast<const unsigned*>(&a_pack1);
    const unsigned a_src0       = __shfl_sync(0xffffffffu, a_pack0_bits, a_src);
    const unsigned a_src1       = __shfl_sync(0xffffffffu, a_pack0_bits, a_src + 1);
    const unsigned a_src2       = __shfl_sync(0xffffffffu, a_pack1_bits, a_src);
    const unsigned a_src3       = __shfl_sync(0xffffffffu, a_pack1_bits, a_src + 1);
    const unsigned a_selector = a_hi ? 0x7632u : 0x5410u;
    const unsigned a_f0_bits  = __byte_perm(a_src0, a_src1, a_selector);
    const unsigned a_f1_bits  = __byte_perm(a_src2, a_src3, a_selector);

    const int b_src   = 8 * (r >> 1) + (r & 1) * 4 + 2 * (t & 1);
    const bool b_hi   = (t >= 2);
    const __half2 b_pack = __floats2half2_rn(fb0, fb1);
    const unsigned b_pack_bits = *reinterpret_cast<const unsigned*>(&b_pack);
    const unsigned b_src0      = __shfl_sync(0xffffffffu, b_pack_bits, b_src);
    const unsigned b_src1      = __shfl_sync(0xffffffffu, b_pack_bits, b_src + 1);
    const unsigned b_selector = b_hi ? 0x7632u : 0x5410u;
    const unsigned b_f0_bits  = __byte_perm(b_src0, b_src1, b_selector);

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a_f0_bits), "r"(a_f1_bits), "r"(b_f0_bits));
#else
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    __trap();
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#else
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    (void)sfa;
    (void)sfb;
    __trap();
#endif
}

} // namespace ninfer::ops
