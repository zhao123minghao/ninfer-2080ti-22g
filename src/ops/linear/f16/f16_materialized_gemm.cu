#include "ops/linear/f16/f16_materialized_gemm.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/launcher/kernel_attr_once.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroupK        = 64;
constexpr int kWordsPerGroup = kGroupK / 8;

constexpr int kQ4CodePg  = Q4RowSplitStorage::kCodeBytesPerGroup;  // 32
constexpr int kQ4ScalePg = Q4RowSplitStorage::kScaleBytesPerGroup; // 2
constexpr int kQ5CodePg  = Q5RowSplitStorage::kCodeBytesPerGroup;  // 32
constexpr int kQ5HighPg  = Q5RowSplitStorage::kHighBytesPerGroup;  // 8
constexpr int kQ5ScalePg = Q5RowSplitStorage::kScaleBytesPerGroup; // 2

// One operand tile. The tile's SHAPE, not its K extent, sets the kernel's cost: staging loads
// `kBlockRows*kBlockK + kBlockCols*kBlockK` elements per K tile to feed
// `kBlockRows*kBlockCols*kBlockK` MACs, so the bytes moved per MAC are `2/kBlockCols` for the weight
// plus `2/kBlockRows` for the activation. An ablation that removed the MMA but kept every load put
// the staging at 9.05 s of the kernel's 20.8 s on a 32K prefill -- 43.6%, against 6.3% for the MMA
// itself (history.md 13).
//
// At 64 rows and 128 columns that is 0.0469 B/MAC with the activation term contributing two
// thirds; 128 rows brings it to 0.03125, and the measured GEMM fell 20.770 -> 17.911 s, which
// matches the predicted -2.98 s to within 0.7% (history.md 14). The wider tile also takes the CTA from
// four warps to eight -- two per sub-partition instead of one -- which that run could not
// attribute separately from the byte reduction.
constexpr int kBlockRows = 128;
constexpr int kBlockCols = 128;
constexpr int kBlockK    = 64;
constexpr int kStages    = 2;
constexpr int kWarpRows  = 64;
constexpr int kWarpCols  = 32;
constexpr int kWarps     = (kBlockRows / kWarpRows) * (kBlockCols / kWarpCols);
constexpr int kThreads   = kWarps * 32;
constexpr int kMmaRows   = kWarpRows / 16;
constexpr int kMmaCols   = kWarpCols / 8;
constexpr int kKSub      = kBlockK / 16;

constexpr int kWgtItems = kBlockRows * (kBlockK / 8) / kThreads;
constexpr int kActItems = kBlockCols * (kBlockK / 8) / kThreads;
static_assert(kWgtItems * kThreads == kBlockRows * (kBlockK / 8));
static_assert(kActItems * kThreads == kBlockCols * (kBlockK / 8));

// The materialization is sliced along output rows: one launch runs a whole slice, and the slice
// width is chosen at run time from the free arena space (see the header). A dequantized slice is
// consumed before the next one is written, so the code/scales planes are read exactly once.
//
// Staging only: the fp16 GEMM has no code/scale/high planes to carry.
constexpr int kSharedBytes =
    kStages * kBlockRows * kBlockK * static_cast<int>(sizeof(__half)) +
    kStages * kBlockCols * kBlockK * static_cast<int>(sizeof(__half));
// At the 128-row tile this is 64 KiB, past the 48 KiB static limit, so the kernel takes its staging
// from dynamic shared memory and the launcher opts into the larger size per device.
static_assert(kSharedBytes <= 64 * 1024, "fp16 GEMM staging exceeds the 64 KiB dynamic budget");

// The XOR swizzle matches the registered kernel; with kBlockK == 64 the group index is exactly the
// eight 16-byte groups of a row, so no mask is needed.
__device__ __forceinline__ int swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __host__ __forceinline__ int div_up_int(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

// The slice width one call may use. Every job of a grouped call shares this figure, because they
// share one materialized buffer.
std::int32_t choose_slice_rows(std::int32_t padded_k, std::int32_t tokens,
                               std::size_t available_bytes) noexcept {
    const std::size_t activation =
        static_cast<std::size_t>(padded_k) * static_cast<std::size_t>(tokens) * sizeof(__half);
    if (available_bytes <= activation) { return 0; }
    const std::size_t per_row = static_cast<std::size_t>(padded_k) * sizeof(__half);
    std::int32_t slice = static_cast<std::int32_t>((available_bytes - activation) / per_row);
    slice -= slice % kBlockRows;
    return slice >= kBlockRows ? slice : 0;
}

// The registered families round their operands with `__floats2bfloat162_rn` and then restage the
// bf16 pair as fp16 on the way into the tensor core. Reproducing that chain -- and not rounding
// straight to fp16 -- is what keeps this route's operand values identical to the registered one.
__device__ __forceinline__ void store_restaged_pair(__half* dst, float w0, float w1) {
    const __nv_bfloat162 rounded = __floats2bfloat162_rn(w0, w1);
    store_vec(dst, bf162_to_f162(*reinterpret_cast<const std::uint32_t*>(&rounded)));
}

// ---------------------------------------------------------------------------------------------
// FP16-accumulating MMA
// ---------------------------------------------------------------------------------------------
//
// Turing's m16n8k8 form admits BOTH `f32.f16.f16.f32` and `f16.f16.f16.f16` accumulators, and they
// do not run at the same rate: measured on this target, the fp32-accumulating form retires
// 512 FLOP/SM/cycle and the fp16-accumulating one 1024 (tools/mma_throughput_probe.cu). That 2x is
// the whole reason a Turing Marlin stack is faster than an fp32-accumulating kernel at the same
// tile, and it is available here without touching a single operand value: the weights and the
// activations stay exactly what the registered route feeds its tensor core.
//
// What it does cost is accumulation precision, and that is why the fp16 accumulator is not the
// operator's accumulator. Each K tile's fp16 partial is folded into the fp32 accumulator that the
// epilogue reads, so the fp16 chain is only ever kBlockK = 64 wide. A 64-long fp16 reduction of
// ~unit-variance terms carries about 64 * 2^-12 = 1.6% of one term's magnitude as error, against a
// partial of sqrt(64) = 8 terms -- roughly 0.2% of the partial, and after the fp32 fold across 80
// groups it stays an order of magnitude inside the A16 criterion's 1/256 relative-L2 allowance
// (tests/ops/linear/linear_test_common.cpp). Accumulating the whole K in fp16 would not: at
// K = 5120 the same analysis gives ~1.2%, which is what made the earlier blanket rejection of
// fp16 accumulation correct for the *unsegmented* form and wrong for this one (history.md 9.5, 11.7).
//
// The fold is nearly free in issue terms. One m16n8k8 occupies the tensor core for 8 cycles per
// sub-partition but only one issue slot, so a 128-MMA K tile leaves ~900 of its 1024 issue slots
// unused; the fold's 128 conversion-plus-add pairs fit in that slack. Measured on the same probe,
// the equivalent epilogue cost 16.5% of an int8 tile's rate, not 50%.
__device__ __forceinline__ void mma_f16_m16n8k8_f16acc(unsigned& c0, unsigned& c1, unsigned a0,
                                                      unsigned a1, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(b0));
}

// The m16n8k16 fragment split, with the fp16 accumulator. The k0..7 product takes (a0, a1) with the
// low half of B and the k8..15 product takes (a2, a3) with the high half -- the same pairing the
// fp32 form uses, and for the same reason (see mma_f16 in ops/common/mma.cuh).
__device__ __forceinline__ void mma_f16_f16acc(unsigned (&c)[2], unsigned a0, unsigned a1,
                                              unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    mma_f16_m16n8k8_f16acc(c[0], c[1], a0, a1, b0);
    mma_f16_m16n8k8_f16acc(c[0], c[1], a2, a3, b1);
}

// Fold one K tile's fp16 partial into the fp32 accumulator the epilogue reads. The two registers
// hold the same four output values, in the same positions, that the fp32 form's four registers do.
__device__ __forceinline__ void fold_f16_partial(float (&dst)[4], unsigned (&src)[2]) {
    const __half2 lo = *reinterpret_cast<const __half2*>(&src[0]);
    const __half2 hi = *reinterpret_cast<const __half2*>(&src[1]);
    const float2 flo = __half22float2(lo);
    const float2 fhi = __half22float2(hi);
    dst[0] += flo.x;
    dst[1] += flo.y;
    dst[2] += fhi.x;
    dst[3] += fhi.y;
    src[0] = 0u;
    src[1] = 0u;
}

// ---------------------------------------------------------------------------------------------
// Materialization
// ---------------------------------------------------------------------------------------------
//
// One 4-byte code word (eight weights) per thread. A whole group per thread would stride the code
// reads by 32 bytes and the fp16 writes by 128 bytes across the warp, which measured 108 GB/s of
// effective traffic against this card's 616 GB/s; word-per-thread coalesces both sides and reaches
// 502 GB/s (history.md 3.11). The scale is shared by the eight threads of a group.
//
// Byte j of a group holds weights 2j and 2j+1, so a little-endian uint32 over bytes 4c..4c+3
// carries weights 8c..8c+7 in order, and the nibble bias trick decodes those eight at once.
__global__ void q4_f16_materialize_kernel(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales,
                                          __half* __restrict__ out, std::int64_t first_row,
                                          std::int64_t rows_in_slice, std::int64_t groups_per_row) {
    const std::int64_t slice_words = rows_in_slice * groups_per_row * kWordsPerGroup;
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= slice_words) { return; }

    const std::int64_t local_group = linear / kWordsPerGroup;
    const int word_in_group        = static_cast<int>(linear - local_group * kWordsPerGroup);
    const std::int64_t group       = first_row * groups_per_row + local_group;

    const float scale = __half2float(
        __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(&scales[group * kQ4ScalePg])));
    const std::uint32_t word =
        load_vec<std::uint32_t>(&codes[group * kQ4CodePg + 4 * word_in_group]) ^ 0x88888888u;

    const __half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032.0
    float weights[8];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
        const __half2 decoded    = __hsub2(half2_from_bits(bits), bias);
        const float2 values      = __half22float2(decoded);
        weights[pair]            = values.x * scale;
        weights[pair + 4]        = values.y * scale;
    }

    __half* dst = &out[local_group * kGroupK + 8 * word_in_group];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        store_restaged_pair(dst + 2 * pair, weights[2 * pair], weights[2 * pair + 1]);
    }
}

// Q5 keeps a second bit plane: high byte c carries bit 4 of weights 8c..8c+7, which the registered
// SIMT atom decodes from the code word, that byte and the group scale.
__global__ void q5_f16_materialize_kernel(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ high,
                                          const std::uint8_t* __restrict__ scales,
                                          __half* __restrict__ out, std::int64_t first_row,
                                          std::int64_t rows_in_slice, std::int64_t groups_per_row) {
    const std::int64_t slice_words = rows_in_slice * groups_per_row * kWordsPerGroup;
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= slice_words) { return; }

    const std::int64_t local_group = linear / kWordsPerGroup;
    const int word_in_group        = static_cast<int>(linear - local_group * kWordsPerGroup);
    const std::int64_t group       = first_row * groups_per_row + local_group;

    const std::uint32_t packed = load_vec<std::uint32_t>(&codes[group * kQ5CodePg + 4 * word_in_group]);
    const std::uint8_t high_bits = high[group * kQ5HighPg + word_in_group];
    const std::uint16_t scale_bits =
        *reinterpret_cast<const std::uint16_t*>(&scales[group * kQ5ScalePg]);

    float weights[8];
    Q5SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);

    __half* dst = &out[local_group * kGroupK + 8 * word_in_group];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        store_restaged_pair(dst + 2 * pair, weights[2 * pair], weights[2 * pair + 1]);
    }
}

// ---------------------------------------------------------------------------------------------
// Activation materialization
// ---------------------------------------------------------------------------------------------
//
// The GEMM's B operand must be fp16, and the hidden state is bf16. Converting it once here costs
// one pass over input_rows * tokens (about 34 us at the registered prefill shapes) and removes the
// per-staged-element conversion from the K-tile loop body, which measured 160 of that body's 513
// instructions (history.md 3.13).
__global__ void bf16_to_f16_kernel(const __nv_bfloat16* __restrict__ in,
                                   __half* __restrict__ out, std::int64_t vectors) {
    const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= vectors) { return; }
    const int4 raw = load_vec<int4>(in + 8 * linear);
    int4 converted;
    converted.x = static_cast<int>(bf162_to_f162(static_cast<unsigned>(raw.x)));
    converted.y = static_cast<int>(bf162_to_f162(static_cast<unsigned>(raw.y)));
    converted.z = static_cast<int>(bf162_to_f162(static_cast<unsigned>(raw.z)));
    converted.w = static_cast<int>(bf162_to_f162(static_cast<unsigned>(raw.w)));
    store_vec(out + 8 * linear, converted);
}

// ---------------------------------------------------------------------------------------------
// Conversion-free fp16 GEMM
// ---------------------------------------------------------------------------------------------
//
// out[Cols, Rows] = W[Rows, padded_k] * x[K, Cols], with both operands already fp16. `out_ld` is
// the destination's leading dimension, so the same kernel serves a standalone projection (where it
// equals the row extent) and one job of a grouped call (where the job's rows sit inside a wider
// destination at an offset).
//
// `Full` is set when the tile divides the geometry exactly, which drops every staging and epilogue
// bound check out of the K-tile loop. The registered kernel carries two instantiations for the same
// reason.
//
// `AddResidual` reproduces the registered residual epilogue exactly: the projection is rounded to
// bf16 first, and only then added to the stored bf16 residual in fp32. Rounding the fp32
// accumulator against the residual directly is a different value, so this two-step form is part of
// the operator's contract.
enum class F16Epilogue { Store, AddResidual };

template <F16Epilogue Ep>
__device__ __forceinline__ void write_out(__nv_bfloat16* dst, float value) {
    if constexpr (Ep == F16Epilogue::Store) {
        *dst = __float2bfloat16_rn(value);
    } else {
        *dst = __float2bfloat16_rn(__bfloat162float(*dst) +
                                  __bfloat162float(__float2bfloat16_rn(value)));
    }
}

template <bool Full, F16Epilogue Ep>
__global__ __launch_bounds__(kThreads) void f16_gemm_kernel(
    const __half* __restrict__ act, const __half* __restrict__ weight,
    __nv_bfloat16* __restrict__ out, std::int32_t rows, std::int32_t k, std::int32_t cols,
    std::int32_t padded_k, std::int32_t slice_rows, std::int32_t row_base,
    std::int32_t out_ld) {
    extern __shared__ __align__(16) __half smem_raw[];
    // [kStages][kBlockRows * kBlockK] followed by [kStages][kBlockCols * kBlockK].
    auto A_at = [&](int s, int off) -> __half* {
        return smem_raw + s * (kBlockRows * kBlockK) + off;
    };
    auto B_at = [&](int s, int off) -> __half* {
        return smem_raw + kStages * (kBlockRows * kBlockK) + s * (kBlockCols * kBlockK) + off;
    };

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int warp_row = warp / (kBlockCols / kWarpCols);
    const int warp_col = warp % (kBlockCols / kWarpCols);

    const int row0 = static_cast<int>(blockIdx.x) * kBlockRows;
    const int col0 = static_cast<int>(blockIdx.y) * kBlockCols;
    const int out_row_base = row_base;

    const int a_matrix     = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    // Staging sources and destinations depend on the thread's item index only, never on the K tile:
    // `row * padded_k` and the swizzled destination offset are K-invariant, and only `k8 * 8 + k0`
    // moves. The registered kernel paid for these once per K tile (history.md 3.7).
    const __half* wgt_src[kWgtItems];
    int wgt_dst[kWgtItems];
#pragma unroll
    for (int i = 0; i < kWgtItems; ++i) {
        const int item = tid + i * kThreads;
        const int row  = item / (kBlockK / 8);
        const int k8   = item - row * (kBlockK / 8);
        wgt_src[i]     = &weight[static_cast<std::int64_t>(row0 + row) * padded_k + k8 * 8];
        wgt_dst[i]     = row * kBlockK + swz(row, k8 * 8);
    }

    const __half* act_src[kActItems];
    int act_dst[kActItems];
#pragma unroll
    for (int i = 0; i < kActItems; ++i) {
        const int item      = tid + i * kThreads;
        const int local_col = item / (kBlockK / 8);
        const int k8        = item - local_col * (kBlockK / 8);
        act_src[i] = &act[static_cast<std::int64_t>(col0 + local_col) * k + k8 * 8];
        act_dst[i] = local_col * kBlockK + swz(local_col, k8 * 8);
    }

    // Turing has no cp.async, so a load+store staging step stalls the issuing warp for a full DRAM
    // round trip. Carry the next tile in registers across the compute instead and touch shared
    // memory only after the current tile has been consumed. At this geometry the block is one CTA
    // per SM with four warps and no co-resident work, so nothing else hides that latency: the
    // standalone probe of history.md 3.22 measures 27.5 -> 33.7 TFLOP/s at M=3072/K=5120 from exactly
    // this change, and the registered kernel without it lands within 4% of the probe's plain path.
    auto load_tile = [&](int kt, int4 (&wgt_pf)[kWgtItems], int4 (&act_pf)[kActItems]) {
        const int k0 = kt * kBlockK;
#pragma unroll
        for (int i = 0; i < kWgtItems; ++i) {
            if (Full || (tid + i * kThreads) / (kBlockK / 8) < slice_rows) {
                wgt_pf[i] = load_ldg<int4>(wgt_src[i] + k0);
            } else {
                wgt_pf[i] = make_int4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int i = 0; i < kActItems; ++i) {
            if (Full || col0 + (tid + i * kThreads) / (kBlockK / 8) < cols) {
                act_pf[i] = load_ldg<int4>(act_src[i] + k0);
            } else {
                act_pf[i] = make_int4(0, 0, 0, 0);
            }
        }
    };

    auto store_tile = [&](int s, const int4 (&wgt_pf)[kWgtItems], const int4 (&act_pf)[kActItems]) {
#pragma unroll
        for (int i = 0; i < kWgtItems; ++i) { store_vec(A_at(s, wgt_dst[i]), wgt_pf[i]); }
#pragma unroll
        for (int i = 0; i < kActItems; ++i) { store_vec(B_at(s, act_dst[i]), act_pf[i]); }
    };

    float accum[kMmaRows][kMmaCols][4];
    // The fp16 partial is the tensor core's accumulator for the current K tile only; `accum` is the
    // operator's accumulator and is what the epilogue reads. Keeping both costs kMmaRows*kMmaCols*2
    // registers on top of the fp32 set.
    unsigned accum_h[kMmaRows][kMmaCols][2];
#pragma unroll
    for (int mi = 0; mi < kMmaRows; ++mi) {
#pragma unroll
        for (int ni = 0; ni < kMmaCols; ++ni) {
#pragma unroll
            for (int c = 0; c < 4; ++c) { accum[mi][ni][c] = 0.0f; }
            accum_h[mi][ni][0] = 0u;
            accum_h[mi][ni][1] = 0u;
        }
    }

    const int k_tiles = padded_k / kBlockK;

#pragma unroll
    for (int s = 0; s < kStages; ++s) {
        if (s < k_tiles) {
            int4 wgt_pf[kWgtItems];
            int4 act_pf[kActItems];
            load_tile(s, wgt_pf, act_pf);
            store_tile(s, wgt_pf, act_pf);
        }
    }
    __syncthreads();

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int s = kt % kStages;

        // Issued before the compute so the DRAM round trip overlaps the 256 HMMA of this K tile.
        int4 wgt_pf[kWgtItems];
        int4 act_pf[kActItems];
        const int prefetch = kt + kStages;
        if (prefetch < k_tiles) { load_tile(prefetch, wgt_pf, act_pf); }

        auto load_fragments = [&](int ki, unsigned (&af)[kMmaRows][4], unsigned (&bf)[kMmaCols][2]) {
            const int k_step = ki * 16;
#pragma unroll
            for (int mi = 0; mi < kMmaRows; ++mi) {
                const int row = warp_row * kWarpRows + mi * 16 + a_row_offset;
                const int col = k_step + a_col_offset;
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                            smem_addr(A_at(s, row * kBlockK + swz(row, col))));
            }
#pragma unroll
            for (int ni = 0; ni < kMmaCols; ++ni) {
                const int row = warp_col * kWarpCols + ni * 8 + b_inner_row;
                const int col = k_step + b_k_offset;
                ldmatrix_x2(bf[ni][0], bf[ni][1],
                            smem_addr(B_at(s, row * kBlockK + swz(row, col))));
            }
        };

#pragma unroll
        for (int ki = 0; ki < kKSub; ++ki) {
            unsigned af[kMmaRows][4];
            unsigned bf[kMmaCols][2];
            load_fragments(ki, af, bf);
#pragma unroll
            for (int mi = 0; mi < kMmaRows; ++mi) {
#pragma unroll
                for (int ni = 0; ni < kMmaCols; ++ni) {
                    mma_f16_f16acc(accum_h[mi][ni], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                   bf[ni][0], bf[ni][1]);
                }
            }
        }

        // Close the fp16 reduction over this K tile. Placed after the whole tile rather than inside
        // the k-step loop so the accumulator is written by eight MMA before anything reads it.
#pragma unroll
        for (int mi = 0; mi < kMmaRows; ++mi) {
#pragma unroll
            for (int ni = 0; ni < kMmaCols; ++ni) {
                fold_f16_partial(accum[mi][ni], accum_h[mi][ni]);
            }
        }

        __syncthreads();
        if (prefetch < k_tiles) { store_tile(s, wgt_pf, act_pf); }
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kMmaRows; ++mi) {
        const int row_a = out_row_base + row0 + warp_row * kWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kMmaCols; ++ni) {
            const int col_a = col0 + warp_col * kWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            const float* values = accum[mi][ni];
            if constexpr (Full) {
                write_out<Ep>(&out[static_cast<std::int64_t>(col_a) * out_ld + row_a], values[0]);
                write_out<Ep>(&out[static_cast<std::int64_t>(col_b) * out_ld + row_a], values[1]);
                write_out<Ep>(&out[static_cast<std::int64_t>(col_a) * out_ld + row_b], values[2]);
                write_out<Ep>(&out[static_cast<std::int64_t>(col_b) * out_ld + row_b], values[3]);
            } else {
                if (row_a < rows) {
                    if (col_a < cols) { write_out<Ep>(&out[static_cast<std::int64_t>(col_a) * out_ld + row_a], values[0]); }
                    if (col_b < cols) { write_out<Ep>(&out[static_cast<std::int64_t>(col_b) * out_ld + row_a], values[1]); }
                }
                if (row_b < rows) {
                    if (col_a < cols) { write_out<Ep>(&out[static_cast<std::int64_t>(col_a) * out_ld + row_b], values[2]); }
                    if (col_b < cols) { write_out<Ep>(&out[static_cast<std::int64_t>(col_b) * out_ld + row_b], values[3]); }
                }
            }
        }
    }
}

} // namespace

bool f16_materialized_supported(const Weight& w, std::int32_t tokens) noexcept {
    // Prefill only: at the decode widths the registered SIMT routes are already free of both the
    // dequantization and the fragment restaging that this route removes, so materializing would be
    // pure overhead.
    if (tokens <= 16) { return false; }
    if (w.qtype != QType::Q4G64_F16S && w.qtype != QType::Q5G64_F16S) { return false; }
    if (w.qdata == nullptr || w.scales == nullptr) { return false; }
    if (w.qtype == QType::Q5G64_F16S && w.qhigh == nullptr) { return false; }
    if (w.n <= 0 || w.k <= 0) { return false; }
    // The K loop runs to the materialized row stride and the decode reads one whole quant group per
    // tile, so the storage must carry no K padding: a padded row would read activation past the
    // logical K for the tail group and decode groups that the packed planes do not hold.
    if (w.padded_shape[1] != w.k) { return false; }
    if (w.k % kGroupK != 0) { return false; }
    // Eight activations per conversion vector, and the operand tile needs whole 16-value MMA steps.
    if ((static_cast<std::int64_t>(w.k) * tokens) % 8 != 0) { return false; }
    return true;
}

std::size_t f16_materialized_workspace_bytes(const Weight& w, std::int32_t tokens,
                                             std::int32_t slice_rows) noexcept {
    if (tokens <= 0 || slice_rows <= 0) { return 0; }
    return static_cast<std::size_t>(slice_rows) * static_cast<std::size_t>(w.k) * sizeof(__half) +
           static_cast<std::size_t>(w.k) * static_cast<std::size_t>(tokens) * sizeof(__half);
}

std::int32_t f16_materialized_choose_slice_rows(const Weight& w, std::int32_t tokens,
                                                std::size_t available_bytes) noexcept {
    if (!f16_materialized_supported(w, tokens)) { return 0; }
    const std::int32_t fits = choose_slice_rows(w.padded_shape[1], tokens, available_bytes);
    if (fits <= 0) { return 0; }
    const std::int32_t rows  = fits < w.n ? fits : w.n;
    const std::int32_t whole = rows - rows % kBlockRows;
    return whole >= kBlockRows ? whole : 0;
}

std::int32_t f16_materialized_slice_rows(const Weight& w, std::int32_t tokens,
                                         const WorkspaceArena& workspace) noexcept {
    return f16_materialized_choose_slice_rows(w, tokens,
                                              workspace.capacity() - workspace.used());
}

namespace {

// One row range of a packed weight decoded into `dst`, which is the slice-local base.
void enqueue_materialize(const Weight& w, std::int32_t first_row, std::int32_t rows,
                         std::int32_t groups_per_row, __half* dst, cudaStream_t stream) {
    constexpr int kMatThreads = 256;
    const std::int64_t words = static_cast<std::int64_t>(rows) *
                               static_cast<std::int64_t>(groups_per_row) * kWordsPerGroup;
    if (words <= 0) { return; }
    const auto blocks = static_cast<unsigned>((words + kMatThreads - 1) / kMatThreads);
    if (w.qtype == QType::Q5G64_F16S) {
        q5_f16_materialize_kernel<<<blocks, kMatThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh),
            static_cast<const std::uint8_t*>(w.scales), dst, first_row, rows, groups_per_row);
    } else {
        q4_f16_materialize_kernel<<<blocks, kMatThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), dst, first_row, rows, groups_per_row);
    }
}

void enqueue_activation(const Tensor& x, __half* dst, cudaStream_t stream) {
    constexpr int kConvertThreads = 256;
    const std::int64_t vectors =
        static_cast<std::int64_t>(x.ne[0]) * static_cast<std::int64_t>(x.ne[1]) / 8;
    if (vectors <= 0) { return; }
    const auto blocks = static_cast<unsigned>((vectors + kConvertThreads - 1) / kConvertThreads);
    bf16_to_f16_kernel<<<blocks, kConvertThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), dst, vectors);
}

// `slice_rows` is the weight-row count this launch covers, starting at weight row 0 of the slice
// buffer; `row_base` and `out_ld` place the result, so the same launch serves a standalone
// projection and one job of a grouped call.
template <F16Epilogue Ep>
void enqueue_gemm(const __half* act, const __half* weight, __nv_bfloat16* out,
                  std::int32_t slice_rows, std::int32_t row_base, std::int32_t out_ld,
                  std::int32_t k, std::int32_t cols, std::int32_t padded_k, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(div_up_int(slice_rows, kBlockRows)),
                    static_cast<unsigned>(div_up_int(cols, kBlockCols)), 1u);
    // `Full` is per launch: the tile must divide both axes exactly for the unguarded path. The
    // guard is the end of this launch's own rows, so a short slice cannot spill into rows another
    // slice owns.
    const bool full        = (slice_rows % kBlockRows == 0) && (cols % kBlockCols == 0);
    const std::int32_t end = row_base + slice_rows;
    if (full) {
        if constexpr (kSharedBytes > 48 * 1024) {
            ensure_func_attr_per_device(f16_gemm_kernel<true, Ep>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
        }
        f16_gemm_kernel<true, Ep><<<grid, kThreads, kSharedBytes, stream>>>(
            act, weight, out, end, k, cols, padded_k, slice_rows, row_base, out_ld);
    } else {
        if constexpr (kSharedBytes > 48 * 1024) {
            ensure_func_attr_per_device(f16_gemm_kernel<false, Ep>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
        }
        f16_gemm_kernel<false, Ep><<<grid, kThreads, kSharedBytes, stream>>>(
            act, weight, out, end, k, cols, padded_k, slice_rows, row_base, out_ld);
    }
    CUDA_CHECK(cudaGetLastError());
}

// One job, sliced along its own rows, reusing the caller's buffer and activation.
template <F16Epilogue Ep>
void run_job(const Tensor& x, const F16GroupedJob& job, std::int32_t padded_k,
             std::int32_t slice_rows, __half* weight_f16, const __half* act_f16,
             cudaStream_t stream) {
    const std::int32_t k              = x.ne[0];
    const std::int32_t cols           = x.ne[1];
    const std::int32_t groups_per_row = padded_k / kGroupK;
    for (std::int32_t row0 = 0; row0 < job.rows; row0 += slice_rows) {
        const std::int32_t this_slice = job.rows - row0 < slice_rows ? job.rows - row0 : slice_rows;
        enqueue_materialize(*job.weight, job.weight_row + row0, this_slice, groups_per_row,
                            weight_f16, stream);
        enqueue_gemm<Ep>(act_f16, weight_f16, job.out, this_slice, job.out_row_offset + row0,
                         job.out_ld, k, cols, padded_k, stream);
    }
}

template <F16Epilogue Ep>
void launch_impl(const Tensor& x, const Weight& w, Tensor& out, std::int32_t slice_rows,
                 WorkspaceArena& workspace, cudaStream_t stream) {
    const F16GroupedJob job{&w, 0, out.ne[0], static_cast<__nv_bfloat16*>(out.data), out.ne[0], 0};
    const std::int32_t padded_k = w.padded_shape[1];

    // The slice buffer is alive only while this call's launches are enqueued, and the arena is
    // shared with every other transient of the phase. Releasing it on return is what lets a later
    // op in the same phase still find room (history.md 3.17).
    auto scope               = workspace.scope();
    const DeviceSpan scratch =
        workspace.alloc_bytes(f16_materialized_workspace_bytes(w, x.ne[1], slice_rows));
    auto* weight_f16 = static_cast<__half*>(scratch.data);
    auto* act_f16 = reinterpret_cast<__half*>(static_cast<std::uint8_t*>(scratch.data) +
                                              static_cast<std::size_t>(slice_rows) *
                                                  static_cast<std::size_t>(padded_k) *
                                                  sizeof(__half));

    enqueue_activation(x, act_f16, stream);
    run_job<Ep>(x, job, padded_k, slice_rows, weight_f16, act_f16, stream);
}

} // namespace

void launch_f16_materialized_gemm(const Tensor& x, const Weight& w, Tensor& out,
                                  std::int32_t slice_rows, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    launch_impl<F16Epilogue::Store>(x, w, out, slice_rows, workspace, stream);
}

void launch_f16_materialized_linear_add(const Tensor& x, const Weight& w, Tensor& residual,
                                        std::int32_t slice_rows, WorkspaceArena& workspace,
                                        cudaStream_t stream) {
    launch_impl<F16Epilogue::AddResidual>(x, w, residual, slice_rows, workspace, stream);
}

bool launch_f16_materialized_grouped(const Tensor& x, const F16GroupedJob* jobs,
                                     std::int32_t job_count, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    if (jobs == nullptr || job_count <= 0 || k <= 0 || cols <= 0) { return false; }

    const std::int32_t padded_k = jobs[0].weight->padded_shape[1];
    // Admit every job before enqueuing any: the call is one atomic decision, because committing a
    // prefix of a grouped projection would leave the caller's registered route reading half-written
    // outputs.
    for (std::int32_t i = 0; i < job_count; ++i) {
        const F16GroupedJob& job = jobs[i];
        if (job.weight == nullptr || job.out == nullptr || job.out_ld <= 0 || job.rows <= 0 ||
            job.weight_row < 0) {
            return false;
        }
        // The jobs of one grouped call share the activation and the padded K, so they share one
        // materialized buffer; anything else keeps the registered route.
        if (job.weight->k != k || job.weight->padded_shape[1] != padded_k) { return false; }
        if (!f16_materialized_supported(*job.weight, cols)) { return false; }
    }
    const std::int32_t slice_rows =
        choose_slice_rows(padded_k, cols, workspace.capacity() - workspace.used());
    if (slice_rows <= 0) { return false; }

    // Same contract as the single-weight entry above: the shared buffer is released on return so the
    // rest of the phase keeps the arena (history.md 3.17).
    auto scope               = workspace.scope();
    const DeviceSpan scratch = workspace.alloc_bytes(
        static_cast<std::size_t>(slice_rows) * static_cast<std::size_t>(padded_k) *
            sizeof(__half) +
        static_cast<std::size_t>(k) * static_cast<std::size_t>(cols) * sizeof(__half));
    auto* weight_f16 = static_cast<__half*>(scratch.data);
    auto* act_f16 = reinterpret_cast<__half*>(static_cast<std::uint8_t*>(scratch.data) +
                                              static_cast<std::size_t>(slice_rows) *
                                                  static_cast<std::size_t>(padded_k) *
                                                  sizeof(__half));

    enqueue_activation(x, act_f16, stream);
    for (std::int32_t i = 0; i < job_count; ++i) {
        run_job<F16Epilogue::Store>(x, jobs[i], padded_k, slice_rows, weight_f16, act_f16, stream);
    }
    return true;
}

} // namespace ninfer::ops::detail
