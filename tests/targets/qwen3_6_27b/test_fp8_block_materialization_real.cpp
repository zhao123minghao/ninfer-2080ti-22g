// Opt-in real-artifact materialization audit for qwen3.8-27b/fp8-block128.
//
// The block-FP8 slice oracle validates the storage primitive in isolation. This test instead
// drives the target's actual bind_artifact(tp=2) plan, materializes its selected FP8 shards on
// both devices, and checks the device bytes against independent reconstruction from each parent
// payload. It closes the target-binder boundary without constructing a Program or running a model.

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using ninfer::artifact::ShardAxis;
using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<std::byte> apply_slice(std::span<const std::byte> parent,
                                   const ninfer::artifact::TensorSlice& slice) {
    std::vector<std::byte> result(static_cast<std::size_t>(slice.encoded_bytes));
    std::copy(slice.prefix.begin(), slice.prefix.end(), result.begin());
    for (const ninfer::artifact::PlaneCopy& copy : slice.copies) {
        require(copy.source_offset + copy.bytes <= parent.size() &&
                    copy.dest_offset + copy.bytes <= result.size(),
                "slice copy escapes its payload");
        std::copy_n(parent.begin() + static_cast<std::ptrdiff_t>(copy.source_offset),
                    static_cast<std::ptrdiff_t>(copy.bytes),
                    result.begin() + static_cast<std::ptrdiff_t>(copy.dest_offset));
    }
    return result;
}

std::vector<std::byte> device_bytes(const ninfer::artifact::MaterializedArtifact& materialized,
                                    ninfer::artifact::ObjectHandle object, int device,
                                    std::uint64_t bytes) {
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpy(result.data(), materialized.device_data(object, device), result.size(),
                          cudaMemcpyDeviceToHost));
    return result;
}

} // namespace

int main() {
    try {
        const char* path = std::getenv("NINFER_FP8_BLOCK_ARTIFACT");
        if (path == nullptr || *path == '\0' || !std::filesystem::is_regular_file(path)) {
            std::cout << "SKIP: NINFER_FP8_BLOCK_ARTIFACT is required\n";
            return 77;
        }
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || device_count < 2) {
            std::cout << "SKIP: two CUDA devices are required\n";
            return 77;
        }
        CUDA_CHECK(status);

        ninfer::artifact::Reader reader(path);
        require(Package::resolve_weights(reader.identity()) == WeightsProfile::Qwen38Fp8Block128,
                "artifact identity is not qwen3.8-27b/fp8-block128");
        ninfer::artifact::Binder binder(reader, 2);
        const ninfer::targets::qwen3_6::StartupFeatures features{};
        const ArtifactLoadPlan load =
            bind_artifact(binder, WeightsProfile::Qwen38Fp8Block128, features, 2);
        const auto& plan = load.materialization;
        require(plan.device_count == 2 && plan.device_capacity_bytes[0] != 0 &&
                    plan.device_capacity_bytes[1] != 0,
                "FP8 TP2 materialization plan is incomplete");

        ninfer::ExecutionContext execution({0, 1});
        const auto materialized = ninfer::artifact::materialize(reader, plan, execution);
        const TextConfig config{};
        const std::array<std::string_view, 7> objects = {
            "text/token_embedding",
            "text/layers/0/gdn/query_key_value_z",
            "text/layers/3/attention/query_key_gate_value",
            "text/layers/0/mlp/gate_up",
            "text/layers/0/mlp/down",
            "text/layers/3/attention/output",
            "text/output_head",
        };

        for (const std::string_view name : objects) {
            std::array<const ninfer::artifact::DeviceMaterialization*, 2> placements{};
            for (const auto& placement : plan.device_objects) {
                if (ninfer::artifact::object_name(reader.objects()[placement.object.index]) == name) {
                    require(placement.device >= 0 && placement.device < 2,
                            "target binder produced an out-of-range device placement");
                    placements[static_cast<std::size_t>(placement.device)] = &placement;
                }
            }
            require(placements[0] != nullptr && placements[1] != nullptr,
                    std::string(name) + " is not placed on both TP2 devices");
            const auto& object = reader.objects()[placements[0]->object.index];
            const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
            require(tensor != nullptr, std::string(name) + " is not a tensor");
            const auto payload = reader.payload(object);
            const ShardMapping mapping = shard_mapping_for(
                name, 2, config, WeightsProfile::Qwen38Fp8Block128);

            for (int device = 0; device < 2; ++device) {
                std::vector<std::byte> expected;
                if (mapping.axis == ShardAxis::Replicated) {
                    expected.assign(payload.data.begin(), payload.data.end());
                } else {
                    std::vector<ninfer::artifact::SliceRange> ranges;
                    for (const Shard& shard : mapping.shards) {
                        if (shard.device == device) {
                            ranges.push_back({shard.row_begin, shard.row_count});
                        }
                    }
                    require(!ranges.empty(), std::string(name) + " has no target shard range");
                    const auto slice = mapping.axis == ShardAxis::Rows
                        ? ninfer::artifact::tensor_row_slice(tensor->layout, tensor->format,
                                                             tensor->shape, ranges, payload.data)
                        : ninfer::artifact::tensor_column_slice(tensor->layout, tensor->format,
                                                                tensor->shape, ranges, payload.data);
                    expected = apply_slice(payload.data, slice);
                }
                const auto* placement = placements[static_cast<std::size_t>(device)];
                require(placement->bytes == expected.size(),
                        std::string(name) + " has the wrong TP2 placement size");
                require(device_bytes(materialized, placement->object, device, placement->bytes) == expected,
                        std::string(name) + " device bytes differ from target TP2 slice");
            }
        }
        std::cout << "real FP8 block128 target TP2 materialization: 7 object families passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}