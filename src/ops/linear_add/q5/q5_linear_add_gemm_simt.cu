#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Cols, int FullSlabs, int Stride>
void launch_split2(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    constexpr int kThreads = 2 * 32;
    const dim3 grid(static_cast<unsigned>(residual_out.ne[0]), 1u, 1u);
    q5_rowsplit_gemm_simt_split2_kernel<Q5RowSplitSimtSchedule, Cols, FullSlabs, Stride, false, 0,
                                        true><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0], x.ne[0], x.ne[1],
        w.padded_shape[1], FullSlabs);
}

template <int Cols>
void dispatch_shape(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (w.k == 6144) {
        launch_split2<Cols, 6, 6144>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_split2<Cols, 17, 17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add split2: unsupported exact K");
    }
}

template <class Launch>
void dispatch_cols(std::int32_t cols, Launch&& launch) {
    switch (cols) {
#define NINFER_Q5_LINEAR_ADD_EXACT(COLS)                                                           \
    case COLS:                                                                                     \
        launch.template operator()<COLS>();                                                        \
        return
        NINFER_Q5_LINEAR_ADD_EXACT(2);
        NINFER_Q5_LINEAR_ADD_EXACT(3);
        NINFER_Q5_LINEAR_ADD_EXACT(4);
        NINFER_Q5_LINEAR_ADD_EXACT(5);
        NINFER_Q5_LINEAR_ADD_EXACT(6);
        NINFER_Q5_LINEAR_ADD_EXACT(7);
        NINFER_Q5_LINEAR_ADD_EXACT(8);
        NINFER_Q5_LINEAR_ADD_EXACT(9);
        NINFER_Q5_LINEAR_ADD_EXACT(10);
        NINFER_Q5_LINEAR_ADD_EXACT(11);
        NINFER_Q5_LINEAR_ADD_EXACT(12);
        NINFER_Q5_LINEAR_ADD_EXACT(13);
        NINFER_Q5_LINEAR_ADD_EXACT(14);
        NINFER_Q5_LINEAR_ADD_EXACT(15);
        NINFER_Q5_LINEAR_ADD_EXACT(16);
#undef NINFER_Q5_LINEAR_ADD_EXACT
    default:
        throw std::invalid_argument("q5 linear_add split2: T must be in [2,16]");
    }
}

} // namespace

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    dispatch_cols(x.ne[1], [&]<int Cols>() { dispatch_shape<Cols>(x, w, residual_out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

// Volta fallback for wide T (the MmaResidualR64C* schedules need Ampere+ mma/ldmatrix, trap-
// stubbed on sm_70): q5_rowsplit_gemm_simt_kernel already takes cols as a runtime grid
// parameter (unlike split2's compile-time Cols switch above), so a single instantiation covers
// any T. AddResidual reuses the exact accumulate-before-store pattern split2 already validated.
// See the V100 performance summary.
//
// Also the tp2 row-parallel shard's decode leaf (T == 1 at K = 3072 / 8704), where it replaces the
// 16x-wasteful MmaResidualR64C16 tile. Tile width follows the same rule ops::linear's own q5 SIMT
// table uses (q5_rowsplit_gemm_simt.cu): four columns for the T<=4 regime -- one whole four-column
// tile, the smallest the schedule has -- and eight for the wider T this fallback was written for.
// At T == 1 a four-column tile costs half the wasted per-tile MMA/issue overhead of an eight-column
// one, and the two extents never overlap, so the split costs the original T=14..16 caller nothing.
template <int ColsPerTile>
void launch_simt_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                          cudaStream_t stream) {
    constexpr int kColsPerTile  = ColsPerTile;
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t rows     = residual_out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, kColsPerTile)), 1u);
    // full_slabs>0 enables the staged/vectorized prefetch path instead of routing every group
    // through the scalar tail (direct global reads). Every offset q5_simt_consume_slab derives
    // from x0 (xslab = slab*1024 elems, c*256 elems, lane*8 elems) is a multiple of 16 bytes in
    // bf16 units, so a single runtime check on x.data's own alignment is sufficient to guarantee
    // every subsequent load_vec<uint4> stays 16-byte aligned -- see
    // q5_rowsplit_gemm_simt.cuh's q5_simt_consume_slab. What remains past full_slabs*1024 is
    // swept by the always-correct scalar tail, so K need not be a multiple of 1024 (the tp2
    // row-parallel shard halves, 3072 and 8704, are exactly that case). Falls back to the
    // all-scalar full_slabs=0 path if that alignment check fails. This is the same predicate
    // ops::linear's own q5 SIMT launcher uses (q5_rowsplit_gemm_simt.cu's launch_simt), which is
    // what the shard's residual-free rank already runs.
    const bool staged_safe = (k % 8) == 0 &&
                             (reinterpret_cast<std::uintptr_t>(x.data) % 16) == 0;
    const std::int32_t full_slabs = staged_safe ? (k / 1024) : 0;
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, kColsPerTile, kRowsPerBlock, kStages,
                                 false, 0, true, 1><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(residual_out.data), nullptr, rows, out_ld, k, cols,
        w.padded_shape[1], full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

void q5_linear_add_simt_wide_t_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    // Exact tile widths for the 5..7 column windows. Those widths are reachable: a round is
    // (1 + draft) columns wide, so draft 4..6 lands here, and before this switch they were served
    // by the eight-column tile -- a whole extra tile row charged for a partial one. Measured on the
    // 85k acceptance workload at T=5 (`--draft-tokens 4`), the selected instantiation was
    // kColsPerTile=8 and cost 189.7us/launch against 96.6us/launch for kColsPerTile=4 at T=4, i.e.
    // 1.96x the work for 1.25x the columns (history.md and todo.md's column-tile note). At T=8 and
    // above the eight-column tile is exact again, and T<=4 keeps its four-column tile.
    switch (x.ne[1]) {
    case 5: launch_simt_residual<5>(x, w, residual_out, stream); return;
    case 6: launch_simt_residual<6>(x, w, residual_out, stream); return;
    case 7: launch_simt_residual<7>(x, w, residual_out, stream); return;
    default: break;
    }
    if (x.ne[1] <= 4) {
        launch_simt_residual<4>(x, w, residual_out, stream);
        return;
    }
    launch_simt_residual<8>(x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
