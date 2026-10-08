#include "ops/linear/fp8_block/fp8_block.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr std::uint64_t kBlockRows = 128;
constexpr std::uint64_t kBlockColumns = 128;
constexpr std::uint64_t kPlaneAlignment = 256;

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(std::string(operation) + ": block-FP8 geometry overflows");
    }
    return left * right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return checked_mul((value + alignment - 1) / alignment, alignment, "FP8 weight");
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool contains(const Weight& weight, const void* pointer, std::uint64_t bytes) {
    if (weight.payload == nullptr || pointer == nullptr) { return false; }
    const auto base = reinterpret_cast<std::uintptr_t>(weight.payload);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (address < base) { return false; }
    const std::uint64_t offset = address - base;
    return offset <= weight.payload_bytes && bytes <= weight.payload_bytes - offset;
}

} // namespace

Fp8BlockWeightGeometry validate_fp8_block_weight(const Weight& weight, const char* operation) {
    // The one entry point the op wrappers use accepts either registered physical layout and checks
    // the invariants that layout declares; the owning dispatch still selects the matching kernel.
    if (weight.layout == QuantLayout::MarlinFp8Block128) {
        return validate_marlin_fp8_block_weight(weight, operation);
    }
    if (weight.n <= 0 || weight.k <= 0 || weight.n % kBlockRows != 0 ||
        weight.k % kBlockColumns != 0) {
        throw std::invalid_argument(std::string(operation) +
                                    ": block-FP8 shape must be positive and 128-aligned");
    }

    Fp8BlockWeightGeometry geometry{};
    geometry.m_tiles = static_cast<std::uint32_t>(weight.n / kBlockRows);
    geometry.k_tiles = static_cast<std::uint32_t>(weight.k / kBlockColumns);
    geometry.code_plane_bytes = checked_mul(static_cast<std::uint64_t>(weight.n),
                                            static_cast<std::uint64_t>(weight.k), operation);
    geometry.scale_plane_offset = align_up(geometry.code_plane_bytes, kPlaneAlignment);
    geometry.scale_plane_bytes = checked_mul(
        checked_mul(geometry.m_tiles, geometry.k_tiles, operation), sizeof(std::uint16_t),
        operation);
    geometry.required_payload_bytes = geometry.scale_plane_offset + geometry.scale_plane_bytes;

    const std::int64_t scale_row_bytes = static_cast<std::int64_t>(geometry.k_tiles) * 2;
    if (weight.qtype != QType::FP8_E4M3FN_BLOCK128_BF16S ||
        weight.layout != QuantLayout::BlockScaleM128K128 || weight.scale_dtype != DType::BF16 ||
        weight.group != 128 || weight.group_size != 128 || weight.ndim != 2 ||
        weight.shape[0] != weight.n || weight.shape[1] != weight.k || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || weight.padded_shape[0] != weight.n ||
        weight.padded_shape[1] != weight.k || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1 || weight.scale_ne[0] != geometry.k_tiles ||
        weight.scale_ne[1] != geometry.m_tiles || weight.scale_ne[2] != 1 ||
        weight.scale_ne[3] != 1 || weight.scale_nb[0] != 2 ||
        weight.scale_nb[1] != scale_row_bytes ||
        weight.scale_nb[2] != scale_row_bytes * geometry.m_tiles ||
        weight.scale_nb[3] != scale_row_bytes * geometry.m_tiles || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || weight.payload_bytes < geometry.required_payload_bytes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16) ||
        !contains(weight, weight.qdata, geometry.code_plane_bytes) ||
        !contains(weight, weight.scales, geometry.scale_plane_bytes)) {
        throw std::invalid_argument(std::string(operation) + ": invalid block-FP8 weight");
    }
    return geometry;
}

Fp8BlockWeightGeometry validate_marlin_fp8_block_weight(const Weight& weight,
                                                        const char* operation) {
    if (weight.n <= 0 || weight.k <= 0 || weight.n % kBlockRows != 0 ||
        weight.k % kBlockColumns != 0) {
        throw std::invalid_argument(std::string(operation) +
                                    ": Marlin block-FP8 shape must have N and K multiples of 128");
    }

    Fp8BlockWeightGeometry geometry{};
    geometry.m_tiles = static_cast<std::uint32_t>(weight.n / kBlockRows);
    geometry.k_tiles = static_cast<std::uint32_t>(weight.k / kBlockColumns);
    geometry.code_plane_bytes = checked_mul(static_cast<std::uint64_t>(weight.n),
                                            static_cast<std::uint64_t>(weight.k), operation);
    geometry.scale_plane_offset = align_up(geometry.code_plane_bytes, kPlaneAlignment);
    geometry.scale_plane_bytes = checked_mul(
        checked_mul(static_cast<std::uint64_t>(geometry.m_tiles),
                    static_cast<std::uint64_t>(geometry.k_tiles), operation),
        sizeof(std::uint16_t), operation);
    geometry.required_payload_bytes = geometry.scale_plane_offset + geometry.scale_plane_bytes;

    const std::int64_t scale_row_bytes = static_cast<std::int64_t>(geometry.k_tiles) * 2;
    if (weight.qtype != QType::FP8_E4M3FN_BLOCK128_BF16S ||
        weight.layout != QuantLayout::MarlinFp8Block128 || weight.scale_dtype != DType::BF16 ||
        weight.group != 128 || weight.group_size != 128 || weight.ndim != 2 ||
        weight.shape[0] != weight.n || weight.shape[1] != weight.k || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || weight.padded_shape[0] != weight.n ||
        weight.padded_shape[1] != weight.k || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1 || weight.scale_ne[0] != geometry.k_tiles ||
        weight.scale_ne[1] != geometry.m_tiles || weight.scale_ne[2] != 1 ||
        weight.scale_ne[3] != 1 || weight.scale_nb[0] != 2 ||
        weight.scale_nb[1] != scale_row_bytes ||
        weight.scale_nb[2] != scale_row_bytes * geometry.m_tiles ||
        weight.scale_nb[3] != scale_row_bytes * geometry.m_tiles || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || weight.payload_bytes < geometry.required_payload_bytes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16) ||
        !contains(weight, weight.qdata, geometry.code_plane_bytes) ||
        !contains(weight, weight.scales, geometry.scale_plane_bytes)) {
        throw std::invalid_argument(std::string(operation) + ": invalid Marlin block-FP8 weight");
    }
    return geometry;
}

std::size_t fp8_block_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                     std::int32_t input_rows,
                                                     LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (output_rows <= 0 || input_rows <= 0 || output_rows % kBlockRows != 0 ||
        input_rows % kBlockColumns != 0 || min_tokens <= 0 || max_tokens < min_tokens ||
        policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("block-FP8 linear workspace: unsupported profile");
    }
    return 0;
}

} // namespace ninfer::ops::detail