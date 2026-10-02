#pragma once

// Q4G64 RowSplit x BF16 SIMT GEMM.
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// One warp owns one output row across ColsPerTile columns. Raw Q4 codes and
// adjacent FP16 scale pairs are staged per row with cp.async; decoded FP32
// weights are reused across the column tile and accumulated with FP32 FMA.
//
// K is staged in whole quant groups. A predicated final stage still copies
// complete 32-byte code groups and aligned 4-byte pairs of FP16 scales. It
// never falls back to scalar code-pair loads, and lanes belonging to inactive
// groups do not form or read an activation address.

#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

template <int RowsPerCta_, int ColsPerTile_, int GroupsPerStage_, int PipelineStages_,
          Cache CodeCache_, int LaunchBoundsMinBlocks_, int RowsPerWarp_ = 1>
struct Q4RowSplitSimtGemmSchedule {
    static_assert(RowsPerWarp_ >= 1 && RowsPerCta_ % RowsPerWarp_ == 0,
                  "the CTA's rows must divide evenly over its warps");
    static constexpr int kRowsPerCta            = RowsPerCta_;
    static constexpr int kColsPerTile           = ColsPerTile_;
    static constexpr int kGroupsPerStage        = GroupsPerStage_;
    static constexpr int kPipelineStages        = PipelineStages_;
    static constexpr Cache kCodeCache           = CodeCache_;
    static constexpr int kLaunchBoundsMinBlocks = LaunchBoundsMinBlocks_;

    // One warp owns `kRowsPerWarp` output rows and feeds a *single* activation read to all of
    // them. Every output row needs the whole x row, so at one row per warp the CTA re-reads a
    // stage's K-slice once per row and the activation stream costs kRowsPerWarp-fold the weight
    // stream -- measured at 327.7 KB against 23.0 KB per CTA. Raising kRowsPerWarp divides that,
    // and it is the only parameter that does: kColsPerTile does not (a row still needs every
    // column), and re-serving the same reads out of shared memory does not either, because
    // Turing's L1TEX unit serves both (history.md 29-30). The arithmetic is untouched -- every
    // accumulator walks the same phases with the same operands in the same order -- so the result
    // is bit-identical and the acceptance counters are the gate.
    static constexpr int kRowsPerWarp = RowsPerWarp_;
    static constexpr int kCtaWarps    = kRowsPerCta / kRowsPerWarp;
    static constexpr int kThreads     = kCtaWarps * 32;
    static constexpr int kStageK   = kGroupsPerStage * Q4RowSplitStorage::kGroupK;
    static constexpr int kCodeVecsPerStage =
        kGroupsPerStage * Q4RowSplitStorage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
    static constexpr int kScalePairsPerStage = kGroupsPerStage / 2;
    static constexpr int kCodePhases         = (kGroupsPerStage + 3) / 4;
    static constexpr int kSharedBytes =
        kRowsPerCta * kPipelineStages *
        (kGroupsPerStage * Q4RowSplitStorage::kCodeBytesPerGroup +
         kScalePairsPerStage * static_cast<int>(sizeof(std::uint32_t)));

    static_assert(kRowsPerCta > 0 && kRowsPerCta <= 32);
    static_assert(kColsPerTile > 0 && kColsPerTile <= 8);
    static_assert(kGroupsPerStage > 0 && kGroupsPerStage % 2 == 0,
                  "Q4 SIMT stages load aligned pairs of FP16 scales");
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8,
                  "Q4 SIMT cp.async pipeline depth must fit cp_wait");
    static_assert(kLaunchBoundsMinBlocks >= 1);
    static_assert(kThreads <= 1024);
    static_assert(kCodeVecsPerStage * static_cast<int>(sizeof(uint4)) ==
                      kGroupsPerStage * Q4RowSplitStorage::kCodeBytesPerGroup,
                  "Q4 code groups must decompose into complete 16-byte vectors");
    static_assert(kSharedBytes <= 48 * 1024,
                  "Q4 SIMT staged shared memory exceeds the static 48 KiB budget");
};

template <class Schedule>
__device__ __forceinline__ void q4_simt_copy_code(uint4* shared_dst,
                                                  const std::uint8_t* global_src) {
    if constexpr (Schedule::kCodeCache == Cache::cg) {
        cp_async<16, Cache::cg>(shared_dst, global_src);
    } else {
        cp_async<16, Cache::ca>(shared_dst, global_src);
    }
}

template <class Schedule, bool FullStage>
__device__ __forceinline__ void q4_simt_issue_stage(uint4* __restrict__ shared_codes,
                                                    std::uint32_t* __restrict__ shared_scales,
                                                    const std::uint8_t* __restrict__ code_row,
                                                    const std::uint8_t* __restrict__ scale_row,
                                                    int stage, int active_groups, int lane) {
    constexpr int kCodeVecs   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairs = Schedule::kScalePairsPerStage;

    const std::int64_t group0        = static_cast<std::int64_t>(stage) * Schedule::kGroupsPerStage;
    const std::uint8_t* stage_codes  = code_row + group0 * Q4RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* stage_scales = scale_row + group0 * Q4RowSplitStorage::kScaleBytesPerGroup;

    const int active_code_vecs = FullStage ? kCodeVecs
                                           : active_groups * Q4RowSplitStorage::kCodeBytesPerGroup /
                                                 static_cast<int>(sizeof(uint4));
    for (int vec = lane; vec < kCodeVecs; vec += 32) {
        if (FullStage || vec < active_code_vecs) {
            q4_simt_copy_code<Schedule>(&shared_codes[vec],
                                        stage_codes + static_cast<std::int64_t>(vec) * 16);
        } else {
            shared_codes[vec] = uint4{0u, 0u, 0u, 0u};
        }
    }

    const int active_scale_pairs = FullStage ? kScalePairs : active_groups / 2;
    for (int pair = lane; pair < kScalePairs; pair += 32) {
        if (FullStage || pair < active_scale_pairs) {
            cp_async<4>(&shared_scales[pair], stage_scales + static_cast<std::int64_t>(pair) * 4);
        } else {
            shared_scales[pair] = 0u;
        }
    }
    cp_commit();
}

#if defined(NINFER_SM75)
// Turing has no cp.async hardware: cp_async<16> above compiles to a synchronous 16-byte load
// feeding a shared store, so calling issue_stage() before the consume stalls this warp on a full
// global round trip before the consume reaches any of its dequant/FMA work. Split the two halves
// around the consume instead -- issue the loads first, then do the shared stores once the current
// stage has been read -- which is the same relocation that removed the exposed staging round trip
// from the prefill fp16 GEMM and the prefill attention (history.md 3.22 / 3.23). Bit-identical: the
// same bytes land in the same shared slots. sm_80+ keeps its real async copies and the fused form.
template <class Schedule, bool FullStage>
struct Q4SimtStageCarrier {
    static constexpr int kCodeVecs   = Schedule::kCodeVecsPerStage;
    static constexpr int kScalePairs = Schedule::kScalePairsPerStage;
    static_assert(kCodeVecs % 32 == 0, "code vectors must divide evenly over the warp");
    static_assert(kScalePairs <= 32, "at most one scale word per lane per stage");
    uint4 codes[kCodeVecs / 32];
    unsigned scales = 0u;
};

template <class Schedule, bool FullStage>
__device__ __forceinline__ void
q4_simt_load_stage(Q4SimtStageCarrier<Schedule, FullStage>& carry,
                   const std::uint8_t* __restrict__ code_row,
                   const std::uint8_t* __restrict__ scale_row, int stage, int active_groups,
                   int lane) {
    using Carrier            = Q4SimtStageCarrier<Schedule, FullStage>;
    constexpr int kCodeVecs   = Carrier::kCodeVecs;
    constexpr int kScalePairs = Carrier::kScalePairs;

    const std::int64_t group0        = static_cast<std::int64_t>(stage) * Schedule::kGroupsPerStage;
    const std::uint8_t* stage_codes  = code_row + group0 * Q4RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* stage_scales = scale_row + group0 * Q4RowSplitStorage::kScaleBytesPerGroup;
    const int active_code_vecs =
        FullStage ? kCodeVecs
                  : active_groups * Q4RowSplitStorage::kCodeBytesPerGroup /
                        static_cast<int>(sizeof(uint4));
    const int active_scale_pairs = FullStage ? kScalePairs : active_groups / 2;

#pragma unroll
    for (int i = 0; i < kCodeVecs / 32; ++i) {
        const int vec = lane + i * 32;
        carry.codes[i] =
            (FullStage || vec < active_code_vecs)
                ? load_ldg<uint4>(stage_codes + static_cast<std::int64_t>(vec) * 16)
                : uint4{0u, 0u, 0u, 0u};
    }
    carry.scales = 0u;
    if (lane < kScalePairs && (FullStage || lane < active_scale_pairs)) {
        carry.scales = load_ldg<unsigned>(stage_scales + static_cast<std::int64_t>(lane) * 4);
    }
}

template <class Schedule, bool FullStage>
__device__ __forceinline__ void
q4_simt_store_stage(uint4* __restrict__ shared_codes, std::uint32_t* __restrict__ shared_scales,
                    const Q4SimtStageCarrier<Schedule, FullStage>& carry, int lane) {
    using Carrier            = Q4SimtStageCarrier<Schedule, FullStage>;
    constexpr int kCodeVecs   = Carrier::kCodeVecs;
    constexpr int kScalePairs = Carrier::kScalePairs;

#pragma unroll
    for (int i = 0; i < kCodeVecs / 32; ++i) { shared_codes[lane + i * 32] = carry.codes[i]; }
    if (lane < kScalePairs) { shared_scales[lane] = carry.scales; }
}
#endif

// `x_origin` already points at this stage's first K element of column 0 of the tile, and
// `x_col_stride` is the element distance between consecutive columns. That indirection lets the
// same body serve either a direct global-memory read (origin = x + col0*k + stage*kStageK,
// stride = k) or a block-staged shared-memory read (origin = the shared tile, stride = kStageK)
// without duplicating the dequant/FMA math -- see the staging rationale in the kernel below.
template <class Schedule, bool FullStage, bool FullCols>
__device__ __forceinline__ void
q4_simt_consume_stage(const __nv_bfloat16* __restrict__ x_origin, std::int64_t x_col_stride,
                      int active_cols, int active_groups,
                      const uint4* __restrict__ shared_codes,
                      const std::uint32_t* __restrict__ shared_scales, int lane,
                      float (&acc)[Schedule::kRowsPerWarp][Schedule::kColsPerTile]) {
    constexpr int kCols       = Schedule::kColsPerTile;
    constexpr int kRows       = Schedule::kRowsPerWarp;
    constexpr int kCodePhases = Schedule::kCodePhases;
    // The staging pointers address row 0 of this warp's slice; the next row's slice of the same
    // pipeline buffer sits one whole row (all stages) further on.
    constexpr int kCodeRowStride  = Schedule::kPipelineStages * Schedule::kCodeVecsPerStage;
    constexpr int kScaleRowStride = Schedule::kPipelineStages * Schedule::kScalePairsPerStage;

    // Measured and NOT kept: issuing each phase's activation loads one phase ahead, so the L1/L2
    // round trip overlaps the previous phase's dequant+FMA chain. It is bit-identical and the
    // kernel did get 1% faster (q4 SIMT 97.8 -> 96.8 us per launch on the T=1 decode workload),
    // but it costs 28 more registers (47 -> 75, dropping residency from 4 to 3 CTAs/SM) and the end
    // to end 85k decode did not move (48.83 tok/s against a 48.39-48.80 baseline band). That is the
    // evidence that ptxas was *already* soft-pipelining these loads with the 47 registers it chose
    // -- which is also why the earlier MinBlocksPerSm=4 attempt could not make it use more. The
    // remaining 44% the x loads cost is therefore not a warp-level latency-hiding problem; it is the
    // number of loads per output row (48 per warp, all feeding one row here). See history.md E.
#pragma unroll
    for (int phase = 0; phase < kCodePhases; ++phase) {
        const int group        = phase * 4 + (lane >> 3);
        const int stage_groups = FullStage ? Schedule::kGroupsPerStage : active_groups;
        if (group < stage_groups) {
            // Decode every row this warp owns *before* touching x: the activation read is the
            // resource, so it happens once per (phase, column) and feeds all kRows rows.
            float weights[kRows][8];
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const std::uint32_t packed =
                    reinterpret_cast<const std::uint32_t*>(shared_codes + r * kCodeRowStride)
                        [phase * 32 + lane];
                const std::uint32_t scale_pair = shared_scales[r * kScaleRowStride + (group >> 1)];
                const std::uint16_t scale_bits =
                    static_cast<std::uint16_t>(scale_pair >> ((group & 1) * 16));
                Q4SimtDecodeAtom::decode_eight(packed, scale_bits, weights[r]);
            }

            const std::int64_t xk = static_cast<std::int64_t>(phase) * 256 + lane * 8;
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                if (FullCols || col < active_cols) {
                    const uint4 values =
                        load_vec<uint4>(x_origin + static_cast<std::int64_t>(col) * x_col_stride +
                                        xk);
                    const float2 x0 = bf16x2_bits_to_float2(values.x);
                    const float2 x1 = bf16x2_bits_to_float2(values.y);
                    const float2 x2 = bf16x2_bits_to_float2(values.z);
                    const float2 x3 = bf16x2_bits_to_float2(values.w);
#pragma unroll
                    for (int r = 0; r < kRows; ++r) {
                        acc[r][col] = fmaf(weights[r][0], x0.x, acc[r][col]);
                        acc[r][col] = fmaf(weights[r][1], x0.y, acc[r][col]);
                        acc[r][col] = fmaf(weights[r][2], x1.x, acc[r][col]);
                        acc[r][col] = fmaf(weights[r][3], x1.y, acc[r][col]);
                        acc[r][col] = fmaf(weights[r][4], x2.x, acc[r][col]);
                        acc[r][col] = fmaf(weights[r][5], x2.y, acc[r][col]);
                        acc[r][col] = fmaf(weights[r][6], x3.x, acc[r][col]);
                        acc[r][col] = fmaf(weights[r][7], x3.y, acc[r][col]);
                    }
                }
            }
        }
    }
}

struct Q4SimtStoreEpilogue {
    template <bool SplitOutput, int SplitRow, int Cols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail, std::int32_t out_ld,
               std::int32_t out_tail_ld, std::int32_t row, std::int32_t col0,
               std::int32_t active_cols, const float (&values)[Cols]) const {
#pragma unroll
        for (int col = 0; col < Cols; ++col) {
            if (col >= active_cols) { continue; }
            if constexpr (SplitOutput) {
                if (row < SplitRow) {
                    out[static_cast<std::int64_t>(col0 + col) * out_ld + row] =
                        __float2bfloat16(values[col]);
                } else {
                    out_tail[static_cast<std::int64_t>(col0 + col) * out_tail_ld + row - SplitRow] =
                        __float2bfloat16(values[col]);
                }
            } else {
                out[static_cast<std::int64_t>(col0 + col) * out_ld + row] =
                    __float2bfloat16(values[col]);
            }
        }
    }
};

template <class Schedule, bool Full, bool SplitOutput = false, int SplitRow = 0,
          class Epilogue = Q4SimtStoreEpilogue, bool TriggerPdl = false, bool JoinPdl = false>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kLaunchBoundsMinBlocks) void q4_rowsplit_gemm_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                                                  const std::
                                                                      uint8_t* __restrict__ codes,
                                                                  const std::
                                                                      uint8_t* __restrict__ scales,
                                                                  __nv_bfloat16* __restrict__ out,
                                                                  __nv_bfloat16* __restrict__ out_tail,
                                                                  std::int32_t out_ld,
                                                                  std::int32_t out_tail_ld,
                                                                  std::int32_t rows, std::int32_t k,
                                                                  std::int32_t cols,
                                                                  std::int32_t padded_k,
                                                                  Epilogue epilogue = {}) {
    static_assert(!SplitOutput || SplitRow > 0,
                  "split-output Q4 SIMT requires a positive compile-time seam");

    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }

    constexpr bool kFull              = Full;
    constexpr int kRowsPerCta         = Schedule::kRowsPerCta;
    constexpr int kRowsPerWarp        = Schedule::kRowsPerWarp;
    constexpr int kColsPerTile        = Schedule::kColsPerTile;
    constexpr int kGroupsPerStage     = Schedule::kGroupsPerStage;
    constexpr int kPipelineStages     = Schedule::kPipelineStages;
    constexpr int kPipelinePrefetch   = kPipelineStages - 1;
    constexpr int kCodeVecsPerStage   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairsPerStage = Schedule::kScalePairsPerStage;

    __shared__ __align__(16) uint4 shared_codes[kRowsPerCta][kPipelineStages][kCodeVecsPerStage];
    __shared__ __align__(16)
        std::uint32_t shared_scales[kRowsPerCta][kPipelineStages][kScalePairsPerStage];
#ifdef NINFER_VOLTA_BUILD
    // One stage's K-slice of activations for the whole column tile, shared by every warp in the
    // CTA. kColsPerTile x kStageK x 2B = 16 KiB at the widest (C8) schedule, on top of the
    // ~8.5 KiB of code/scale staging -- still inside the 48 KiB static budget, and free in
    // occupancy terms because this kernel is register-limited (ncu: Block Limit Registers 2)
    // well before shared memory binds.
    //
    // Not enabled for sm_75. It has been ported and measured there twice, and lost both times:
    // 50.04 -> 48.88 tok/s (2.3%) at 99 regs / 2 blocks/SM, and 51.63 -> 49.11 tok/s (4.7%) after
    // the kernel moved to minBlocks 3 (58 regs, 24 warps/SM). Acceptance was bit-identical both
    // times, so this is a pure traffic/pipeline effect and not a numerical one.
    //
    // The second measurement is the informative one, because it was taken *after* an ncu profile
    // had named L1TEX the binding resource -- 89.87% against SM 72.63% and DRAM 42% -- and
    // attributed 93.5% of that L1 traffic to exactly this redundancy: 8 warps x 4 cols x K 5120 x
    // 2B = 327.7 KB of activation reads per CTA against 23.0 KB of weight. The staging really
    // does collapse the global side 8x, and the kernel still got slower. The reason is that
    // shared memory is served by the same L1TEX unit as global memory: staging re-issues every
    // one of those activation reads as a shared load, so the wavefront count barely moves, while
    // three block-wide barriers per stage are added on top of an otherwise warp-local pipeline.
    // What the x-to-constant ablation on this kernel measured was the cost of issuing and waiting
    // on those reads at all -- not of where they are served from.
    //
    // The lever that does reduce them is per-warp reuse: one warp owning r > 1 output rows, so a
    // single activation read feeds r rows. `kCtaWarps` is currently pinned to `kRowsPerCta` (one
    // row per warp), which is what makes the redundancy exactly kRowsPerCta-fold. See
    // history.md 30.
    __shared__ __align__(16) __nv_bfloat16 x_stage[kColsPerTile * Schedule::kStageK];
    static_assert(kColsPerTile * Schedule::kStageK * static_cast<int>(sizeof(__nv_bfloat16)) +
                          Schedule::kSharedBytes <=
                      48 * 1024,
                  "Q4 SIMT activation staging must fit the static shared budget");
#endif

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row0 = static_cast<int>(blockIdx.x) * kRowsPerCta + warp * kRowsPerWarp;
    // A warp whose row is past the end must NOT return early: the activation staging below is a
    // block-wide cooperative step with __syncthreads(), and an early return would deadlock the
    // barrier for the surviving warps. Instead the row is clamped to a valid one (so every load
    // stays in bounds) and only the epilogue store is suppressed.
    int row[kRowsPerWarp];
    bool row_active[kRowsPerWarp];
    int safe_row[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        row[r]        = row0 + r;
        row_active[r] = kFull || row[r] < rows;
        safe_row[r]   = row_active[r] ? row[r] : (rows > 0 ? rows - 1 : 0);
    }

    const int col0        = static_cast<int>(blockIdx.y) * kColsPerTile;
    const int active_cols = kFull ? kColsPerTile : min(kColsPerTile, cols - col0);

    const int padded_groups = padded_k / Q4RowSplitStorage::kGroupK;
    const int groups        = k / Q4RowSplitStorage::kGroupK;
    const int stages =
        kFull ? groups / kGroupsPerStage : (groups + kGroupsPerStage - 1) / kGroupsPerStage;

    const std::uint8_t* code_row[kRowsPerWarp];
    const std::uint8_t* scale_row[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        code_row[r] = codes + static_cast<std::int64_t>(safe_row[r]) * padded_groups *
                                  Q4RowSplitStorage::kCodeBytesPerGroup;
        scale_row[r] = scales + static_cast<std::int64_t>(safe_row[r]) * padded_groups *
                                    Q4RowSplitStorage::kScaleBytesPerGroup;
    }

    float acc[kRowsPerWarp][kColsPerTile];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
        for (int col = 0; col < kColsPerTile; ++col) { acc[r][col] = 0.0f; }
    }

#pragma unroll
    for (int prefetch = 0; prefetch < kPipelinePrefetch; ++prefetch) {
        if (prefetch < stages) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - prefetch * kGroupsPerStage);
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                q4_simt_issue_stage<Schedule, kFull>(
                    shared_codes[warp * kRowsPerWarp + r][prefetch],
                    shared_scales[warp * kRowsPerWarp + r][prefetch], code_row[r], scale_row[r],
                    prefetch, active_groups, lane);
            }
        } else {
            cp_commit();
        }
    }

#pragma unroll 1
    for (int stage = 0; stage < stages; ++stage) {
        const int fetch = stage + kPipelinePrefetch;
        const bool fetching = fetch < stages;
#if defined(NINFER_SM75)
        const int fetch_groups =
            fetching ? (kFull ? kGroupsPerStage
                              : min(kGroupsPerStage, groups - fetch * kGroupsPerStage))
                     : 0;
        Q4SimtStageCarrier<Schedule, kFull> carry[kRowsPerWarp];
        if (fetching) {
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                q4_simt_load_stage<Schedule, kFull>(carry[r], code_row[r], scale_row[r], fetch,
                                                    fetch_groups, lane);
            }
        } else {
            cp_commit();
        }
#else
        if (fetching) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - fetch * kGroupsPerStage);
            const int buffer = fetch % kPipelineStages;
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                q4_simt_issue_stage<Schedule, kFull>(
                    shared_codes[warp * kRowsPerWarp + r][buffer],
                    shared_scales[warp * kRowsPerWarp + r][buffer], code_row[r], scale_row[r],
                    fetch, active_groups, lane);
            }
        } else {
            cp_commit();
        }
#endif

        cp_wait<kPipelinePrefetch>();
        __syncwarp();

        const int active_groups =
            kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - stage * kGroupsPerStage);
        const int buffer = stage % kPipelineStages;
#ifdef NINFER_VOLTA_BUILD
        // Every warp in this CTA owns a different output row but reads the *same* activation
        // slice, so reading x straight from global (as the pre-staging code did, once per column
        // per phase per warp) costs kRowsPerCta-fold redundant L1/LSU traffic -- ~30x the weight
        // traffic at this op's shapes, and the single structural difference between this kernel
        // and the much more efficient T=1 GEMV sibling (which stages x in shared). Staging the
        // stage's K-slice once per CTA collapses that to a single read. bf16 is kept as the
        // staged type so the consume path's uint4 vector loads and its exact bf16->float
        // conversion are both bit-for-bit unchanged -- this is a pure traffic optimisation with
        // no numerical effect, unlike the HFMA2/dp4a experiments recorded in the V100 implementation.
        __syncthreads();
        {
            const int staged_vecs = (active_groups * Q4RowSplitStorage::kGroupK) / 8;
            const std::int64_t src_k0 = static_cast<std::int64_t>(stage) * Schedule::kStageK;
            for (int i = static_cast<int>(threadIdx.x); i < active_cols * staged_vecs;
                 i += static_cast<int>(blockDim.x)) {
                const int col = i / staged_vecs;
                const int j   = i - col * staged_vecs;
                const uint4 v = load_vec<uint4>(
                    x + static_cast<std::int64_t>(col0 + col) * k + src_k0 + j * 8);
                *reinterpret_cast<uint4*>(&x_stage[col * Schedule::kStageK + j * 8]) = v;
            }
        }
        __syncthreads();
        q4_simt_consume_stage<Schedule, kFull, kFull>(x_stage, Schedule::kStageK, active_cols,
                                                      active_groups,
                                                      &shared_codes[warp * kRowsPerWarp][buffer][0],
                                                      &shared_scales[warp * kRowsPerWarp][buffer][0],
                                                      lane, acc);
        __syncthreads();
#else
        q4_simt_consume_stage<Schedule, kFull, kFull>(
            x + static_cast<std::int64_t>(col0) * k + static_cast<std::int64_t>(stage) * Schedule::kStageK,
            k, active_cols, active_groups, &shared_codes[warp * kRowsPerWarp][buffer][0],
            &shared_scales[warp * kRowsPerWarp][buffer][0], lane, acc);
        __syncwarp();
#endif
#if defined(NINFER_SM75)
        if (fetching) {
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                q4_simt_store_stage<Schedule, kFull>(
                    shared_codes[warp * kRowsPerWarp + r][fetch % kPipelineStages],
                    shared_scales[warp * kRowsPerWarp + r][fetch % kPipelineStages], carry[r], lane);
            }
        }
        // The consume reads the scale slots written by the low lanes (group >> 1), so the next
        // iteration's read needs a warp-visible publish of this stage's stores.
        __syncwarp();
#endif
    }

    if constexpr (std::is_same_v<Epilogue, Q4SimtStoreEpilogue>) {
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
            for (int col = 0; col < kColsPerTile; ++col) {
                if (kFull || col < active_cols) {
                    const float sum = warp_reduce_sum(acc[r][col]);
                    if (lane == 0 && row_active[r]) {
                        if constexpr (SplitOutput) {
                            if (row[r] < SplitRow) {
                                out[static_cast<std::int64_t>(col0 + col) * out_ld + row[r]] =
                                    __float2bfloat16(sum);
                            } else {
                                out_tail[static_cast<std::int64_t>(col0 + col) * out_tail_ld + row[r] -
                                         SplitRow] = __float2bfloat16(sum);
                            }
                        } else {
                            out[static_cast<std::int64_t>(col0 + col) * out_ld + row[r]] =
                                __float2bfloat16(sum);
                        }
                    }
                }
            }
        }
    } else {
        float sums[kRowsPerWarp][kColsPerTile];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
            for (int col = 0; col < kColsPerTile; ++col) {
                sums[r][col] = (kFull || col < active_cols) ? warp_reduce_sum(acc[r][col]) : 0.0F;
            }
        }
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (lane == 0 && row_active[r]) {
                epilogue.template operator()<SplitOutput, SplitRow>(out, out_tail, out_ld, out_tail_ld,
                                                                    row[r], col0, active_cols,
                                                                    sums[r]);
            }
        }
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

} // namespace ninfer::ops::detail
