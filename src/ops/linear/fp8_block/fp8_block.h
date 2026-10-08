#pragma once

#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8BlockWeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
    std::uint32_t m_tiles;
    std::uint32_t k_tiles;
};

[[nodiscard]] Fp8BlockWeightGeometry validate_fp8_block_weight(const Weight& weight,
                                                               const char* operation);
[[nodiscard]] Fp8BlockWeightGeometry validate_marlin_fp8_block_weight(const Weight& weight,
                                                                       const char* operation);

[[nodiscard]] std::size_t fp8_block_linear_workspace_capacity_bytes(
    std::int32_t output_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens);

void fp8_block_linear_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream);
void fp8_block_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream);
void fp8_block_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);
void fp8_block_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& query,
                                  Tensor& gate, Tensor& key, Tensor& value,
                                  cudaStream_t stream);
void fp8_block_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail