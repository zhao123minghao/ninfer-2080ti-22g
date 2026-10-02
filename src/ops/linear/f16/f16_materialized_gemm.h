#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// FP16 materialization of a packed GroupwiseInt row-split weight plus its activation, consumed by
// a conversion-free fp16 GEMM.
//
// Why this exists
// ---------------
// Turing has no bf16 tensor core, so the registered MMA route restages every bf16 fragment into
// fp16 on the way into the tensor core, and it dequantizes inside the K loop. Measured against a
// dequantization-free fp16 GEMM at the identical tile shape (64x128xBK=64, four warps), the
// registered kernel issues 793 instructions per K tile where the clean kernel issues 329 -- both
// produce 128 HMMA. That difference is the codes/scales staging, the dequantization and the
// fragment restaging (history.md 3.9-3.13).
//
// Numerics
// --------
// Weight: `bf162_to_f162(bits of __floats2bfloat162_rn(code * scale))`. Activation:
// `bf162_to_f162(bits of the bf16 activation)`. Both are exactly what the registered route feeds
// its tensor core -- the Q4 nibble-bias decode and the Q5 five-bit decode here are the registered
// families' own atoms -- so this is a relocation of work, not a change of arithmetic, and the
// operator's output must stay bit-identical.
//
// Scope
// -----
// Prefill only. The decode shapes (T <= 16) resolve to the SIMT routes, where the materialization
// would dominate the GEMM it feeds.

// True when this weight and token extent can take the materialized route at all, ignoring how much
// arena happens to be free.
[[nodiscard]] bool f16_materialized_supported(const Weight& w, std::int32_t tokens) noexcept;

// Bytes the materialization needs for this weight at this token extent and slice width: the fp16
// weight slice plus the fp16 activation.
[[nodiscard]] std::size_t f16_materialized_workspace_bytes(const Weight& w, std::int32_t tokens,
                                                           std::int32_t slice_rows) noexcept;

// Largest row slice of this weight that fits in `available_bytes` alongside the fp16 activation,
// rounded down to a whole tile, or 0 when the route does not apply or does not fit.
//
// The slice exists because linear's workspace is a shared arena other transients occupy while a
// GEMM runs; a full materialization of the widest Q4 prefill weight is about 180 MiB and did not
// fit (history.md 3.13). It is sized from the free space rather than fixed, because the first
// implementation's fixed 4096-row slice is not the largest the arena allows.
[[nodiscard]] std::int32_t f16_materialized_choose_slice_rows(const Weight& w, std::int32_t tokens,
                                                             std::size_t available_bytes) noexcept;

// `choose_slice_rows` against the arena's own free space: the slice width to use, or 0 to decline.
[[nodiscard]] std::int32_t f16_materialized_slice_rows(const Weight& w, std::int32_t tokens,
                                                       const WorkspaceArena& workspace) noexcept;

// Materialize the weight and the activation into the arena and run the GEMM against them.
void launch_f16_materialized_gemm(const Tensor& x, const Weight& w, Tensor& out,
                                  std::int32_t slice_rows, WorkspaceArena& workspace,
                                  cudaStream_t stream);

// The same route with the registered residual epilogue: `residual` holds the residual on entry and
// receives the sum. `rows` is read from `residual`.
void launch_f16_materialized_linear_add(const Tensor& x, const Weight& w, Tensor& residual,
                                        std::int32_t slice_rows, WorkspaceArena& workspace,
                                        cudaStream_t stream);

// One projection inside a grouped call: a row view of a packed weight together with the place its
// output goes. The grouped input projections stack several such views into one logical Op call, so
// they share the activation and differ only in rows and destination.
struct F16GroupedJob {
    const Weight* weight        = nullptr;
    std::int32_t weight_row     = 0;  // first source row of this view inside `weight`
    std::int32_t rows           = 0;  // this view's row count
    __nv_bfloat16* out          = nullptr;
    std::int32_t out_ld         = 0;  // destination leading dimension
    std::int32_t out_row_offset = 0;  // destination row offset
};

// Materialize every job's weight view and the shared activation, then run one conversion-free fp16
// GEMM per job. Returns false -- enqueuing nothing -- when the call is not admitted, so the caller
// keeps its registered route. Every job must consume the same activation and share one padded K.
[[nodiscard]] bool launch_f16_materialized_grouped(const Tensor& x, const F16GroupedJob* jobs,
                                                   std::int32_t job_count,
                                                   WorkspaceArena& workspace,
                                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
