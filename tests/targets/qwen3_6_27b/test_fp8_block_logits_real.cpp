// Opt-in FP8 block128 versus groupwise-int logit probe.
//
// This is a diagnostic boundary test, not a quality oracle: it compares registered artifacts or
// scalar/HMMA routes under the same eager TP2 execution profile. It reports complete-vocabulary
// BF16 logit distance, top-8 candidates, and layer-boundary differences.

#include "ninfer/engine.h"
#include "artifact/reader.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kContext = 512;
constexpr std::uint32_t kOutputTokens = 15;
constexpr std::uint32_t kRouteComparePrefillChunk = 4096;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kHidden = 5120;

using Tokens = std::vector<ninfer::TokenId>;
using Logits = std::vector<std::uint16_t>;
using DevicePair = std::array<int, 2>;

struct Capture {
    Tokens tokens;
    Logits position_zero;
    std::vector<std::uint16_t> position_zero_layers;
};

struct BoundaryCapture {
    Logits logits;
    std::vector<std::uint16_t> layers;
};

struct TopEntry {
    ninfer::TokenId token = -1;
    float value = -std::numeric_limits<float>::infinity();
};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

class ScopedFp8Route {
public:
    explicit ScopedFp8Route(const char* route) {
        const char* previous = std::getenv("NINFER_FP8_BLOCK_ROUTE");
        had_previous_ = previous != nullptr;
        if (had_previous_) { previous_value_ = previous; }
        if (route == nullptr) {
            require(unsetenv("NINFER_FP8_BLOCK_ROUTE") == 0,
                    "unable to clear NINFER_FP8_BLOCK_ROUTE");
        } else {
            require(setenv("NINFER_FP8_BLOCK_ROUTE", route, 1) == 0,
                    "unable to set NINFER_FP8_BLOCK_ROUTE");
        }
    }

    ~ScopedFp8Route() {
        if (had_previous_) {
            setenv("NINFER_FP8_BLOCK_ROUTE", previous_value_.c_str(), 1);
        } else {
            unsetenv("NINFER_FP8_BLOCK_ROUTE");
        }
    }

private:
    bool had_previous_ = false;
    std::string previous_value_;
};

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

Tokens read_tokens(const char* path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("unable to open NINFER_FP8_BLOCK_PROMPT_IDS"); }
    Tokens tokens;
    std::int64_t token = 0;
    while (input >> token) {
        require(token >= 0 && token <= std::numeric_limits<ninfer::TokenId>::max(),
                "prompt token is outside TokenId range");
        tokens.push_back(static_cast<ninfer::TokenId>(token));
    }
    require(!tokens.empty(), "prompt token file is empty");
    return tokens;
}

DevicePair read_device_pair(const char* variable) {
    const char* value = std::getenv(variable);
    require(value != nullptr, std::string(variable) + " is required");
    const std::string text(value);
    const std::size_t comma = text.find(',');
    require(comma != std::string::npos && text.find(',', comma + 1) == std::string::npos,
        std::string(variable) + " must be formatted as device0,device1");
    const int first = std::stoi(text.substr(0, comma));
    const int second = std::stoi(text.substr(comma + 1));
    require(first >= 0 && second >= 0 && first != second,
        std::string(variable) + " must name two distinct non-negative devices");
    return {first, second};
}

std::uint32_t read_positive_u32(const char* variable, std::uint32_t fallback) {
    const char* value = std::getenv(variable);
    if (value == nullptr) { return fallback; }
    const unsigned long parsed = std::stoul(value);
    require(parsed > 0 && parsed <= std::numeric_limits<std::uint32_t>::max(),
            std::string(variable) + " must be a positive uint32 value");
    return static_cast<std::uint32_t>(parsed);
}

ninfer::EngineOptions engine_options(const char* artifact, DevicePair devices,
                                     std::uint32_t max_context = kContext,
                                     std::uint32_t prefill_chunk = 1024) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = max_context;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.prefill_chunk = prefill_chunk;
    options.kv_cache = ninfer::KvCacheStorage::Float16;
    options.tp = 2;
    options.devices = {devices[0], devices[1]};
    options.use_cuda_graph = false;
    return options;
}

ninfer::RequestOptions request_options(std::uint32_t output_tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = output_tokens;
    options.execution.allow_prefix_reuse = false;
    options.execution.sampling.temperature = 0.0F;
    options.stop.include_model_defaults = false;
    return options;
}

void verify_embedding(const char* artifact, const Tokens& context,
                      const BoundaryCapture& capture) {
    ninfer::artifact::Reader reader(artifact);
    if (reader.identity().weights_id != "fp8-block128") { return; }
    const auto* object = reader.find("text/token_embedding");
    require(object != nullptr, "artifact has no token embedding");
    const auto* descriptor = std::get_if<ninfer::artifact::TensorDescriptor>(object);
    require(descriptor != nullptr &&
                descriptor->format == ninfer::artifact::NumericFormat::BF16 &&
                descriptor->layout == ninfer::artifact::StorageLayout::ContiguousLeV1 &&
                descriptor->shape.size() == 2 && descriptor->shape[1] == kHidden,
            "FP8 artifact embedding is not a contiguous BF16 matrix");
    const ninfer::TokenId token = context.back();
    require(token >= 0 && static_cast<std::uint64_t>(token) < descriptor->shape[0],
            "last prompt token is outside the embedding vocabulary");
    const auto payload = reader.payload(*object).data;
    const std::size_t row_bytes = kHidden * sizeof(std::uint16_t);
    const std::size_t offset = static_cast<std::size_t>(token) * row_bytes;
    require(offset + row_bytes <= payload.size() && capture.layers.size() >= kHidden,
            "embedding oracle or capture has an invalid extent");
    for (std::size_t channel = 0; channel < kHidden; ++channel) {
        std::uint16_t expected = 0;
        std::memcpy(&expected, payload.data() + offset + channel * sizeof(expected),
                    sizeof(expected));
        require(capture.layers[channel] == expected,
                "embedding capture mismatch: token=" + std::to_string(token) +
                    " channel=" + std::to_string(channel) +
                    " expected_word=" + std::to_string(expected) +
                    " captured_word=" + std::to_string(capture.layers[channel]));
    }
    std::cout << "embedding oracle exact: token=" << token << " words=" << kHidden << '\n';
}

BoundaryCapture generate_and_capture(ninfer::Engine& engine, const Tokens& context,
                                     std::uint32_t output_tokens, Tokens* generated = nullptr) {
    const ninfer::GenerationResult result = engine.generate(
        engine.prepare_tokens(context, /*allow_prefix_identity=*/false), request_options(output_tokens));
    require(result.generated_token_ids.size() == output_tokens,
            "logit probe did not generate its requested token count");
    require(result.reused_prompt_tokens == 0, "logit probe unexpectedly reused a prefix");
    if (generated != nullptr) { *generated = result.generated_token_ids; }
    BoundaryCapture capture{engine.debug_last_round_logits_bf16(),
                            engine.debug_last_prefill_layers_bf16()};
    require(!capture.logits.empty(), "logit capture did not produce a vocabulary vector");
    require(!capture.layers.empty() && capture.layers.size() % kHidden == 0,
            "layer capture did not produce complete hidden vectors");
    return capture;
}

Capture capture(const char* artifact, const Tokens& prompt, DevicePair devices,
                std::uint32_t max_context = kContext, std::uint32_t prefill_chunk = 1024,
                const char* route = nullptr) {
    ScopedFp8Route route_scope(route);
    ninfer::Engine engine(engine_options(artifact, devices, max_context, prefill_chunk));
    engine.debug_enable_logit_capture(true);
    engine.debug_enable_layer_capture(true);
    Capture out;
    // Capture occurs at prefill finalization, so this is position zero even though the request
    // continues through ordinary decode. A later decode position needs a new prefill whose
    // prompt explicitly includes the preceding generated tokens.
    const BoundaryCapture boundary = generate_and_capture(engine, prompt, kOutputTokens, &out.tokens);
    verify_embedding(artifact, prompt, boundary);
    out.position_zero = boundary.logits;
    out.position_zero_layers = boundary.layers;
    require(out.tokens.front() == static_cast<ninfer::TokenId>(
                std::max_element(out.position_zero.begin(), out.position_zero.end(),
                                 [](std::uint16_t a, std::uint16_t b) {
                                     return bf16_to_float(a) < bf16_to_float(b);
                                 }) - out.position_zero.begin()),
            "position-zero captured logit argmax differs from the sampled token");
    return out;
}

BoundaryCapture capture_context(const char* artifact, const Tokens& context, DevicePair devices,
                                std::uint32_t max_context = kContext,
                                std::uint32_t prefill_chunk = 1024) {
    ninfer::Engine engine(engine_options(artifact, devices, max_context, prefill_chunk));
    engine.debug_enable_logit_capture(true);
    engine.debug_enable_layer_capture(true);
    const BoundaryCapture boundary = generate_and_capture(engine, context, 1);
    verify_embedding(artifact, context, boundary);
    return boundary;
}

std::vector<TopEntry> top_k(const Logits& logits) {
    std::vector<TopEntry> entries;
    entries.reserve(logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const float value = bf16_to_float(logits[index]);
        require(std::isfinite(value), "captured logits contain a non-finite value");
        entries.push_back({static_cast<ninfer::TokenId>(index), value});
    }
    const auto middle = entries.begin() + static_cast<std::ptrdiff_t>(std::min(kTopK, entries.size()));
    std::partial_sort(entries.begin(), middle, entries.end(),
                      [](const TopEntry& a, const TopEntry& b) { return a.value > b.value; });
    entries.resize(std::min(kTopK, entries.size()));
    return entries;
}

std::size_t shared_prefix(const Tokens& first, const Tokens& second) {
    std::size_t count = 0;
    while (count < first.size() && count < second.size() && first[count] == second[count]) {
        ++count;
    }
    return count;
}

void report(std::size_t position, const Logits& first, const Logits& second,
            const char* first_label = "fp8", const char* second_label = "groupwise") {
    require(first.size() == second.size(), "captures have different vocabulary extents");
    double delta_squared = 0.0;
    double reference_squared = 0.0;
    double maximum_absolute = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        const double a = bf16_to_float(first[index]);
        const double b = bf16_to_float(second[index]);
        const double difference = a - b;
        delta_squared += difference * difference;
        reference_squared += b * b;
        maximum_absolute = std::max(maximum_absolute, std::abs(difference));
    }
    const auto first_top = top_k(first);
    const auto second_top = top_k(second);
    const auto margin = [](const std::vector<TopEntry>& top) {
        return top.size() > 1 ? top[0].value - top[1].value : 0.0F;
    };
    std::cout << std::setprecision(8) << "logit position=" << position
              << " max_abs=" << maximum_absolute
              << " rel_l2=" << std::sqrt(delta_squared / std::max(reference_squared, 1.0))
              << ' ' << first_label << "_top1=" << first_top.front().token
              << ' ' << first_label << "_margin=" << margin(first_top)
              << ' ' << second_label << "_top1=" << second_top.front().token
              << ' ' << second_label << "_margin=" << margin(second_top) << '\n';
    for (const auto& [label, top] : std::array<std::pair<const char*, std::vector<TopEntry>>, 2>{
             std::pair{first_label, first_top}, std::pair{second_label, second_top}}) {
        std::cout << "  " << label << " top" << top.size() << ':';
        for (const TopEntry& entry : top) {
            std::cout << ' ' << entry.token << ':' << entry.value;
        }
        std::cout << '\n';
    }
}

void report_layers(std::size_t position, const std::vector<std::uint16_t>& first,
           const std::vector<std::uint16_t>& second,
           const char* first_label = "fp8", const char* second_label = "groupwise") {
    require(first.size() == second.size() && first.size() % kHidden == 0,
            "artifacts captured different layer-boundary extents");
    const std::size_t boundaries = first.size() / kHidden;
    require(boundaries >= 2, "layer capture is missing the embedding or post-mixer boundaries");
    const bool summary_only = std::getenv("NINFER_FP8_BLOCK_SUMMARY_ONLY") != nullptr;
    std::cout << "layer boundaries position=" << position << " post_mixer_count=" << boundaries - 1
              << '\n';
    for (std::size_t boundary = 0; boundary < boundaries; ++boundary) {
        if (summary_only && boundary != 0 && boundary != 1 && boundary + 1 != boundaries) {
            continue;
        }
        double delta_squared = 0.0;
        double reference_squared = 0.0;
        double maximum_absolute = 0.0;
        const std::size_t offset = boundary * kHidden;
        for (std::size_t index = 0; index < kHidden; ++index) {
            const double a = bf16_to_float(first[offset + index]);
            const double b = bf16_to_float(second[offset + index]);
            const double difference = a - b;
            delta_squared += difference * difference;
            reference_squared += b * b;
            maximum_absolute = std::max(maximum_absolute, std::abs(difference));
        }
        std::cout << "  " << first_label << "/" << second_label << ' '
              << (boundary == 0 ? "embedding" : "post_mixer")
              << "=" << (boundary == 0 ? 0 : boundary - 1)
              << " max_abs=" << maximum_absolute
                  << " rel_l2=" << std::sqrt(delta_squared / std::max(reference_squared, 1.0))
                  << '\n';
    }
}

} // namespace

int main() {
    try {
        const char* fp8_artifact = std::getenv("NINFER_FP8_BLOCK_ARTIFACT");
        const char* groupwise_artifact = std::getenv("NINFER_GROUPWISE_INT_ARTIFACT");
        const char* prompt_path = std::getenv("NINFER_FP8_BLOCK_PROMPT_IDS");
        const bool route_compare = std::getenv("NINFER_FP8_BLOCK_ROUTE_COMPARE") != nullptr;
        if (fp8_artifact == nullptr || prompt_path == nullptr ||
            (!route_compare && groupwise_artifact == nullptr)) {
            std::cout << "SKIP: NINFER_FP8_BLOCK_ARTIFACT and NINFER_FP8_BLOCK_PROMPT_IDS are "
                         "required; groupwise artifact is required unless route compare is enabled\n";
            return 77;
        }
        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver ||
            device_count < 2) {
            std::cout << "SKIP: two CUDA devices are required\n";
            return 77;
        }
        if (device_status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(device_status));
        }

        const Tokens prompt = read_tokens(prompt_path);
        const char* device_text = std::getenv("NINFER_FP8_BLOCK_DEVICES");
        const DevicePair device_pair = device_text == nullptr
            ? DevicePair{0, 1}
            : read_device_pair("NINFER_FP8_BLOCK_DEVICES");
        if (route_compare) {
            const std::uint32_t max_context = std::max<std::uint32_t>(
                kContext, static_cast<std::uint32_t>(prompt.size() + kOutputTokens));
            const std::uint32_t default_chunk = read_positive_u32(
                "NINFER_FP8_BLOCK_PREFILL_CHUNK", kRouteComparePrefillChunk);
            const std::uint32_t chunk_a = read_positive_u32(
                "NINFER_FP8_BLOCK_PREFILL_CHUNK_A", default_chunk);
            const std::uint32_t chunk_b = read_positive_u32(
                "NINFER_FP8_BLOCK_PREFILL_CHUNK_B", chunk_a);
            const char* route_a = std::getenv("NINFER_FP8_BLOCK_ROUTE_A");
            const char* route_b = std::getenv("NINFER_FP8_BLOCK_ROUTE_B");
            if (route_a == nullptr) { route_a = "scalar"; }
            if (route_b == nullptr) { route_b = "hmma"; }
            const Capture scalar = capture(fp8_artifact, prompt, device_pair, max_context, chunk_a,
                                           route_a);
            const Capture hmma = capture(fp8_artifact, prompt, device_pair, max_context, chunk_b,
                                         route_b);
            const std::size_t common = shared_prefix(scalar.tokens, hmma.tokens);
            std::cout << "same-artifact route compare " << route_a << " vs " << route_b
                      << " shared greedy prefix=" << common;
            if (common < scalar.tokens.size() && common < hmma.tokens.size()) {
                std::cout << " first_divergence=" << common << " scalar=" << scalar.tokens[common]
                          << " hmma=" << hmma.tokens[common];
            }
            std::cout << '\n';
            report(0, scalar.position_zero, hmma.position_zero, route_a, route_b);
            report_layers(0, scalar.position_zero_layers, hmma.position_zero_layers,
                          route_a, route_b);
            return 0;
        }

        const char* pair_a_text = std::getenv("NINFER_FP8_BLOCK_DEVICE_PAIR_A");
        const char* pair_b_text = std::getenv("NINFER_FP8_BLOCK_DEVICE_PAIR_B");
        if ((pair_a_text == nullptr) != (pair_b_text == nullptr)) {
            throw std::runtime_error(
                "NINFER_FP8_BLOCK_DEVICE_PAIR_A and _B must be provided together");
        }
        if (pair_a_text != nullptr) {
            const DevicePair pair_a = read_device_pair("NINFER_FP8_BLOCK_DEVICE_PAIR_A");
            const DevicePair pair_b = read_device_pair("NINFER_FP8_BLOCK_DEVICE_PAIR_B");
            const std::uint32_t max_context = std::max<std::uint32_t>(
                kContext, static_cast<std::uint32_t>(prompt.size() + kOutputTokens));
            const std::uint32_t chunk_a = read_positive_u32(
                "NINFER_FP8_BLOCK_PREFILL_CHUNK_A", kRouteComparePrefillChunk);
            const std::uint32_t chunk_b = read_positive_u32(
                "NINFER_FP8_BLOCK_PREFILL_CHUNK_B", chunk_a);
            const Capture first = capture(fp8_artifact, prompt, pair_a, max_context, chunk_a);
            const Capture second = capture(fp8_artifact, prompt, pair_b, max_context, chunk_b);
            require(first.tokens == second.tokens,
                    "same-artifact TP2 device pairs generated different token ids");
            std::cout << "same-artifact TP2 device-pair comparison passed token equality\n";
            report(0, first.position_zero, second.position_zero, "pair_a", "pair_b");
            report_layers(0, first.position_zero_layers, second.position_zero_layers,
                          "pair_a", "pair_b");
            return 0;
        }

        const Capture fp8 = capture(fp8_artifact, prompt, device_pair);
        const Capture groupwise = capture(groupwise_artifact, prompt, device_pair);
        const std::size_t common = shared_prefix(fp8.tokens, groupwise.tokens);
        std::cout << "shared greedy prefix=" << common;
        if (common < fp8.tokens.size() && common < groupwise.tokens.size()) {
            std::cout << " first_divergence=" << common << " fp8=" << fp8.tokens[common]
                      << " groupwise=" << groupwise.tokens[common];
        }
        std::cout << '\n';
        report(0, fp8.position_zero, groupwise.position_zero);
        report_layers(0, fp8.position_zero_layers, groupwise.position_zero_layers);
        if (common == kOutputTokens - 1) {
            Tokens context = prompt;
            context.insert(context.end(), fp8.tokens.begin(),
                           fp8.tokens.begin() + static_cast<std::ptrdiff_t>(common));
            const BoundaryCapture fp8_boundary = capture_context(fp8_artifact, context, device_pair);
            const BoundaryCapture groupwise_boundary =
                capture_context(groupwise_artifact, context, device_pair);
            report(common, fp8_boundary.logits, groupwise_boundary.logits);
            report_layers(common, fp8_boundary.layers, groupwise_boundary.layers);
        } else {
            std::cout << "last-position logit vectors are not context-aligned; shared prefix is "
                      << common << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}