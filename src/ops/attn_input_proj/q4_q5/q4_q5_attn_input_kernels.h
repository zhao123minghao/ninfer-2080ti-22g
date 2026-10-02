#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

// tp2 column-shard sibling of the small-T route above (query/gate 3072, key/value 512). Compile-
// time-exact to the shard's halved extents; admits T == 1 only (the decode leaf).
void q4_q5_attn_input_small_t_shard_launch(const Tensor& x, const Weight& query_key_weight,
                                           const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                           Tensor& k, Tensor& v, cudaStream_t stream);

void q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    WorkspaceArena* workspace,
                                                    cudaStream_t stream);

void q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    WorkspaceArena* workspace,
                                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
