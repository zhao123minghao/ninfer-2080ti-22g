#pragma once

// ninfer::ops - signed int8, per-token group-wise KV cache codec (shared device
// helpers). Quantization (append) and dequantization (stage) are FUSED into the
// GQA attention kernels themselves (decode partial kernel, prefill fill/attention);
// this header only provides the index math, the vectorized dequant, and the scalar
// quantize helper they share. There is deliberately no standalone quant/dequant
// kernel: that would defeat the halved-bandwidth goal.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaKvQuantHeadDim = 256;
inline constexpr int kGqaKvQuantGroup   = 64;
inline constexpr int kGqaKvQuantGroups  = kGqaKvQuantHeadDim / kGqaKvQuantGroup;

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_scale_index(int physical_page, int kv_head,
                                                                 int group, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantGroups, Geometry::KVHeads>(physical_page, kv_head,
                                                                         page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaKvQuantHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

// Quantize one bf16 value with a precomputed 1/scale (scale is the FP16-rounded
// per-group absmax/127). Round-to-nearest-even + symmetric clamp to keep codes
// bit-identical to the CPU oracle and to bf16 parity.
__device__ __forceinline__ std::int8_t gqa_kv_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

// Dequantize 8 consecutive int8 codes (dims [d, d+8), aligned to a multiple of 8
// so they lie inside one 64-group) into 8 bf16 packed as an int4, given a pointer
// to the 8 codes and the group's dequant scale. The codes are read with ONE 64-bit
// (int2) load; the pointer may be in global or shared memory. This keeps the dequant
// ALU identical whether the codes were streamed via cp.async into smem (decode) or
// read directly from the cache (prefill).
__device__ __forceinline__ int4 gqa_kv_dequant_i8x8_from(const std::int8_t* codes8, float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Same contract as gqa_kv_dequant_i8x8_from, but packs fp16 instead of bf16: Volta's
// mma.sync.m8n8k4 accepts fp16 operands only, so the tensor-core decode path needs its staged
// K/V in fp16, while the SIMT path wants bf16 to match its accumulators. Still one 64-bit load.
__device__ __forceinline__ int4 gqa_kv_dequant_i8x8_f16_from(const std::int8_t* codes8, float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    __half2 packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed[i] = __floats2half2_rn(static_cast<float>(c[2 * i]) * s,
                                      static_cast<float>(c[2 * i + 1]) * s);
    }
    return *reinterpret_cast<const int4*>(packed);
}

// --- fp8 (E4M3) KV codec ----------------------------------------------------------------
//
// The fp8 KV cache stores the bf16 K/V activations as E4M3 codes with no per-group scale plane
// (the E4M3 exponent already covers the KV activation range). Quantization is one bf16->E4M3
// cast at append; dequantization is the same ALU bit transform the weight path uses (FP8 exp plane
// shifted one bit into the fp16 position, then x256 to restore the exponent bias), which is exact
// for normals and subnormals and maps the 0x7f/0xff NaN payloads to +/-Inf.

__device__ __forceinline__ std::uint8_t gqa_kv_quant_fp8(float x) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// Quantize 8 consecutive bf16 activations (16 bytes, one int4) into 8 packed fp8 codes
// (8 bytes, one int2), in d order. Mirrors the bf16 prefill fill kernel's 8-element vector
// granularity so the fp8 fill kernel writes one cache-aligned 8-byte vector per element.
__device__ __forceinline__ int2 gqa_kv_quant_fp8x8_bf16(const int4 raw_bf16x8) {
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw_bf16x8);
    int2 out;
    std::uint8_t* o = reinterpret_cast<std::uint8_t*>(&out);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        o[i] = gqa_kv_quant_fp8(__bfloat162float(b[i]));
    }
    return out;
}

__device__ __forceinline__ __half2 gqa_kv_dequant_fp8x2_f16(std::uint16_t packed) {
    const unsigned q = packed;
    constexpr unsigned kMask = 0x7F007F00u;
    const unsigned first = (q & 0x80008000u) | ((q & kMask) >> 1);
    const unsigned second = ((q << 8) & 0x80008000u) | (((q << 8) & kMask) >> 1);
    const unsigned combined = (second & 0xFFFFu) | ((first & 0xFFFFu) << 16);
    const __half2 shifted = *reinterpret_cast<const __half2*>(&combined);
    return __hmul2(shifted, __half2half2(__ushort_as_half(0x5C00u)));  // x 2^8
}

// Dequantize 8 consecutive fp8 codes into 8 fp16 packed as an int4 (one 64-bit load). No scale
// plane: the fp8 storage is the only plane. Volta/Turing mma.sync takes fp16 operands.
__device__ __forceinline__ int4 gqa_kv_dequant_fp8x8_f16_from(const std::uint8_t* codes8) {
    const int2 raw            = load_vec<int2>(codes8);
    const std::uint8_t* c     = reinterpret_cast<const std::uint8_t*>(&raw);
    __half2 packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint16_t pair = static_cast<std::uint16_t>(c[2 * i]) |
                                   (static_cast<std::uint16_t>(c[2 * i + 1]) << 8);
        packed[i] = gqa_kv_dequant_fp8x2_f16(pair);
    }
    return *reinterpret_cast<const int4*>(packed);
}

} // namespace ninfer::ops
