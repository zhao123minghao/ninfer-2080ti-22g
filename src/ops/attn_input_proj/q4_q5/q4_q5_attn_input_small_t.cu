#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kParentRows = 7168;
constexpr std::int32_t kSplitRow   = 6144;
constexpr std::int32_t kHidden     = 5120;

using Q4AttnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 3>;
using Q4AttnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 3>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule, true, kSplitRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(key.data), kParentRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, true, kSplitRow>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(key.data), q.ne[0], key.ne[0], kParentRows, kHidden, cols,
            weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                          cudaStream_t stream) {
    const bool full = (kParentRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Schedule, true>(x, weight, q, key, stream);
    } else {
        launch_q4_simt<Schedule, false>(x, weight, q, key, stream);
    }
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv(x, weight, q, key, stream);
        return;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
        launch_q4_simt_route<Q4AttnSimtR8C4Schedule>(x, weight, q, key, stream);
        return;
    case 8:
    case 16:
        launch_q4_simt_route<Q4AttnSimtR8C8Schedule>(x, weight, q, key, stream);
        return;
    default:
#ifdef NINFER_VOLTA_BUILD
        // GroupedHomogeneousPairMma* (T>16) need Ampere+ mma/ldmatrix, trap-stubbed on sm_70.
        // launch_q4_simt_route already takes cols as a runtime grid parameter (the switch above
        // is only picking a tile-size schedule for tuning, not a kernel limit), so it
        // generalizes to any T unchanged. See the V100 performance summary.
        if (x.ne[1] > 16) {
            launch_q4_simt_route<Q4AttnSimtR8C8Schedule>(x, weight, q, key, stream);
            return;
        }
#endif
        throw std::invalid_argument("attention Q4 split-output requires T in [1,16]");
    }
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kGrid         = kParentRows / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kParentRows, kHidden, kRowsPerBlock, 2, true, false, true, kSplitRow>
        <<<kGrid, kBlockThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              static_cast<const std::uint8_t*>(weight.qdata),
                                              static_cast<const std::uint8_t*>(weight.qhigh),
                                              static_cast<const std::uint8_t*>(weight.scales),
                                              static_cast<__nv_bfloat16*>(gate.data),
                                              static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                      cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(kParentRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, kHidden, true, kSplitRow>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(gate.data),
                                        static_cast<__nv_bfloat16*>(value.data), kParentRows,
                                        gate.ne[0], kHidden, Cols, weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, gate, value, stream);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, gate, value, stream);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,6]");
    }
}

template <int ColsPerTile>
void launch_q5_simt(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages, true,
                                 kSplitRow><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(value.data), kParentRows, gate.ne[0], kHidden, cols,
        weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 6) {
        launch_q5_split4_exact(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        launch_q5_simt<4>(x, weight, gate, value, stream);
        return;
    }
#ifdef NINFER_VOLTA_BUILD
    // Same reasoning as launch_q4 above: launch_q5_simt<4> already takes cols as a runtime grid
    // parameter, so it generalizes past T=16 once GroupedHomogeneousPairMma* is unavailable.
    launch_q5_simt<4>(x, weight, gate, value, stream);
    return;
#else
    throw std::invalid_argument("attention Q5 split-output requires T in [1,16]");
#endif
}

} // namespace

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_q4(x, query_key_weight, q, k, stream);
    launch_q5(x, gate_value_weight, gate, v, stream);
}

// --- tp2 column shard (query/gate 3072, key/value 512; fused parent 3584) -----------------------
//
// The tp1 small-T launchers above are compile-time-exact to the parent's 7168/6144 rows. The shard
// is the SAME two fused split-output kernels at the halved row counts: the Q4 GEMV reads `rows`
// from the launch and takes its split seam as a compile-time constant, and the Q5 GEMV is
// templated on both. Only the two extents change, so the shard gets its own thin launchers rather
// than ever returning to the 64-column grouped MMA tile (right for prefill, but it pays a whole
// 64x128 tile per decode column). Decode T=1 is the entire point of these two.
namespace {

constexpr std::int32_t kShardParentRows = 3584; // query 3072 + key 512
constexpr std::int32_t kShardSplitRow   = 3072;

void launch_shard_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                          cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kShardParentRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule, true, kShardSplitRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(key.data), kShardParentRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

void launch_shard_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                          cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kGrid         = kShardParentRows / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kShardParentRows, kHidden, kRowsPerBlock, 2, true, false, true,
                            kShardSplitRow><<<kGrid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

// T in [2,6]: the shard's own instantiations of the split-output SIMT pair the tp1
// ParentSplitFixed route uses (see launch_q4_simt / launch_q5_split4 above). Every extents
// argument that the tp1 launchers take from kParentRows/kSplitRow is taken from the shard's
// kShardParentRows/kShardSplitRow instead; the leading-dimension arguments (q.ne[0], key.ne[0],
// gate.ne[0]) and the runtime column count are passed exactly as the tp1 forms pass them. The
// SIMT kernels read cols from the launch, so one instantiation per tile schedule covers the whole
// range. The upper edge is the widest decode round this build admits. It is NOT 1 + draft: a
// round is formed over every active request, so its width is concurrency * (1 + draft), and each
// small-T path below owns the range it can serve at one instantiation per real column. The
// single-request figure (6) sent every multi-request round straight to the grouped 32x64 prefill
// tile, which pays ~64 columns of MMA work for eight real ones -- measured as a 3x per-call cost
// on the decode round (history.md 22.3).
constexpr std::int32_t kShardMaxCols = 12;

template <class Schedule, bool Full>
void launch_shard_q4_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                          cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kShardParentRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, true, kShardSplitRow>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(key.data), q.ne[0],
            key.ne[0], kShardParentRows, kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_shard_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                                cudaStream_t stream) {
    using Schedule = Q4AttnSimtR8C4Schedule;
    const bool full = (kShardParentRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_shard_q4_simt<Schedule, true>(x, weight, q, key, stream);
    } else {
        launch_shard_q4_simt<Schedule, false>(x, weight, q, key, stream);
    }
}

template <int Cols>
void launch_shard_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    // Two rows per CTA: this Op is at 92% of the L1 ceiling and x is the same column block for
    // every output row, so one shared x decode per two rows is the whole lever (history.md §39).
    constexpr int kRowsPerCta = 2;
    constexpr int kGrid       = (kShardParentRows + kRowsPerCta - 1) / kRowsPerCta;
    const dim3 grid(static_cast<unsigned>(kGrid), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, kHidden, true,
                                        kShardSplitRow, Q5Split4StoreEpilogue, false, false,
                                        kRowsPerCta><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(value.data),
        kShardParentRows, gate.ne[0], kHidden, Cols, weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_shard_q5_simt(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                          cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_shard_q5_split4<2>(x, weight, gate, value, stream);
        return;
    case 3:
        launch_shard_q5_split4<3>(x, weight, gate, value, stream);
        return;
    case 4:
        launch_shard_q5_split4<4>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_shard_q5_split4<5>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_shard_q5_split4<6>(x, weight, gate, value, stream);
        return;
    case 7:
        launch_shard_q5_split4<7>(x, weight, gate, value, stream);
        return;
    case 8:
        launch_shard_q5_split4<8>(x, weight, gate, value, stream);
        return;
    case 9:
        launch_shard_q5_split4<9>(x, weight, gate, value, stream);
        return;
    case 10:
        launch_shard_q5_split4<10>(x, weight, gate, value, stream);
        return;
    case 11:
        launch_shard_q5_split4<11>(x, weight, gate, value, stream);
        return;
    case 12:
        launch_shard_q5_split4<12>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 shard split4 requires T in [2,12]");
    }
}

} // namespace

void q4_q5_attn_input_small_t_shard_launch(const Tensor& x, const Weight& query_key_weight,
                                           const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                           Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_shard_q4_gemv(x, query_key_weight, q, k, stream);
        launch_shard_q5_gemv(x, gate_value_weight, gate, v, stream);
        return;
    }
    if (x.ne[1] <= kShardMaxCols) {
        launch_shard_q4_simt_route(x, query_key_weight, q, k, stream);
        launch_shard_q5_simt(x, gate_value_weight, gate, v, stream);
        return;
    }
    throw std::invalid_argument("attention Q4/Q5 shard small-T launch requires T in [1,6]");
}

} // namespace ninfer::ops::detail
