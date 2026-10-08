// Engine-side half of the source-aligned quality oracle for the block-FP8 .ninfer artifact.
//
// `tools/parity/qwen3_8_27b/dump_source_logits.py` exports the source checkpoint's next-token
// top-k for one fixed token-ID context; this tool exports the same quantity from the public
// Engine over the same ids, in a JSON shape that can be compared position by position.
//
// Each probe is a fresh, single-token request over one prefix of the ids file. A budget of one
// token completes on prefill's own sampled token, which is exactly where
// `debug_enable_logit_capture()` publishes the full-vocabulary vector (see engine.h). `--positions
// N` probes the last N prefixes (lengths `len - N + 1 .. len`), i.e. the teacher-forced walk used
// to find the first position where Engine and source disagree.
//
// usage:
//   engine_logits_probe --weights <artifact.ninfer> --ids-file <ids.txt> --out <report.json>
//                       [--positions N] [--top-k K] [--tp 1|2] [--devices 0,1]
//                       [--prefill-chunk N] [--kv-dtype fp16|int8]
//
// Exits 77 (ctest "skip") when --weights or --ids-file is absent, so an unconditional ctest run
// on a host without the local artifact does not fail.

#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

// Engine logits are published as raw BF16 bit patterns, in token-id order.
float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value              = 0.0F;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

std::vector<ninfer::TokenId> read_ids(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot read ids file: " + path.string()); }
    std::vector<ninfer::TokenId> ids;
    long long value = 0;
    while (input >> value) {
        if (value < 0) { throw std::runtime_error("ids must be non-negative"); }
        ids.push_back(static_cast<ninfer::TokenId>(value));
    }
    if (ids.empty()) { throw std::runtime_error("ids file is empty: " + path.string()); }
    return ids;
}

struct Options {
    std::filesystem::path weights;
    std::filesystem::path ids;
    std::filesystem::path out;
    int tp                      = 2;
    std::vector<int> devices    = {0, 1};
    std::uint32_t positions     = 1;
    std::uint32_t top_k         = 8;
    std::uint32_t prefill_chunk = 1024;
    ninfer::KvCacheStorage kv   = ninfer::KvCacheStorage::Float16;
};

[[noreturn]] void usage(const std::string& problem) {
    std::cerr << "engine_logits_probe: " << problem << "\n"
              << "usage: engine_logits_probe --weights <artifact.ninfer> --ids-file <ids.txt> "
                 "--out <report.json>\n"
                 "                           [--positions N] [--top-k K] [--tp 1|2] "
                         "[--devices 0,1]\n"
                         "                           [--prefill-chunk N] [--kv-dtype fp16|int8]\n";
    std::exit(1);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next            = [&](const char* label) -> std::string {
            if (++index >= argc) { usage(std::string("missing value for ") + label); }
            return argv[index];
        };
        if (argument == "--weights") {
            options.weights = next("--weights");
        } else if (argument == "--ids-file") {
            options.ids = next("--ids-file");
        } else if (argument == "--out") {
            options.out = next("--out");
        } else if (argument == "--positions") {
            options.positions = static_cast<std::uint32_t>(std::stoul(next("--positions")));
        } else if (argument == "--top-k") {
            options.top_k = static_cast<std::uint32_t>(std::stoul(next("--top-k")));
        } else if (argument == "--tp") {
            options.tp = std::stoi(next("--tp"));
        } else if (argument == "--devices") {
            options.devices.clear();
            std::string list = next("--devices");
            std::size_t begin = 0;
            while (begin < list.size()) {
                const std::size_t end = list.find(',', begin);
                options.devices.push_back(std::stoi(
                    list.substr(begin, end == std::string::npos ? std::string::npos : end - begin)));
                if (end == std::string::npos) { break; }
                begin = end + 1;
            }
        } else if (argument == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(std::stoul(next("--prefill-chunk")));
        } else if (argument == "--kv-dtype") {
            const std::string value = next("--kv-dtype");
            if (value == "fp16") {
                options.kv = ninfer::KvCacheStorage::Float16;
            } else if (value == "int8") {
                options.kv = ninfer::KvCacheStorage::Int8Group64;
            } else {
                usage("--kv-dtype must be fp16 or int8, got " + value);
            }
        } else if (argument == "--help" || argument == "-h") {
            usage("help requested");
        } else {
            usage("unknown argument " + argument);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    if (argc <= 1) {
        std::cerr << "engine_logits_probe: no arguments; skipping\n";
        return 77;
    }
    const Options options = parse_options(argc, argv);
    if (options.weights.empty() || options.ids.empty() || options.out.empty()) {
        std::cerr << "engine_logits_probe: --weights, --ids-file and --out are all required; "
                     "skipping\n";
        return 77;
    }
    if (!std::filesystem::exists(options.weights)) {
        std::cerr << "engine_logits_probe: artifact not found: " << options.weights << "; skipping\n";
        return 77;
    }
    if (options.tp < 1 || options.tp > 2 ||
        options.devices.size() != static_cast<std::size_t>(options.tp)) {
        usage("--devices must name exactly --tp distinct devices");
    }
    if (options.positions < 1 || options.top_k < 1) {
        usage("--positions and --top-k must be positive");
    }

    const std::vector<ninfer::TokenId> ids = read_ids(options.ids);
    if (options.positions > ids.size()) {
        usage("--positions exceeds the number of ids (" + std::to_string(ids.size()) + ")");
    }

    ninfer::EngineOptions engine_options;
    engine_options.artifact_path  = options.weights;
    engine_options.device         = options.devices.front();
    engine_options.tp             = options.tp;
    engine_options.devices        = options.devices;
    engine_options.max_context    = static_cast<std::uint32_t>(ids.size()) + 8;
    engine_options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(
        static_cast<std::uint32_t>(ids.size()) + 8);
    engine_options.prefill_chunk  = options.prefill_chunk;
    engine_options.kv_cache       = options.kv;
    engine_options.use_cuda_graph = false;

    ninfer::Engine engine(engine_options);
    engine.debug_enable_logit_capture(true);

    Json probes = Json::array();
    const std::uint32_t first =
        static_cast<std::uint32_t>(ids.size()) - options.positions + 1;
    for (std::uint32_t length = first; length <= ids.size(); ++length) {
        std::vector<ninfer::TokenId> context(ids.begin(), ids.begin() + length);
        ninfer::PreparedPrompt prepared = engine.prepare_tokens(context, false);

        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 1;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = false;

        const ninfer::GenerationResult result = engine.generate(std::move(prepared), request);
        const std::vector<std::uint16_t> bits = engine.debug_last_round_logits_bf16();
        if (bits.empty()) {
            throw std::runtime_error("logit capture returned no values at context length " +
                                     std::to_string(length));
        }

        std::vector<float> logits;
        logits.reserve(bits.size());
        for (const std::uint16_t value : bits) { logits.push_back(bf16_to_float(value)); }

        std::vector<std::uint32_t> order(logits.size());
        for (std::uint32_t index = 0; index < order.size(); ++index) { order[index] = index; }
        const std::uint32_t keep = std::min<std::uint32_t>(options.top_k,
                                                           static_cast<std::uint32_t>(order.size()));
        std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                          [&](std::uint32_t left, std::uint32_t right) {
                              return logits[left] > logits[right];
                          });

        Json top = Json::array();
        for (std::uint32_t rank = 0; rank < keep; ++rank) {
            top.push_back(Json{{"token_id", order[rank]},
                               {"logit", logits[order[rank]]},
                               {"rank", rank + 1}});
        }
        const double margin = keep >= 2 ? static_cast<double>(logits[order[0]]) -
                                              static_cast<double>(logits[order[1]])
                                        : 0.0;
        const ninfer::TokenId sampled =
            result.generated_token_ids.empty() ? order[0] : result.generated_token_ids.front();

        probes.push_back(Json{{"position", length},
                              {"context_tokens", length},
                              {"sampled_token_id", sampled},
                              {"argmax_token_id", order[0]},
                              {"margin", margin},
                              {"top", std::move(top)}});
        std::cout << "  probe length " << length << ": argmax " << order[0] << " (sampled "
                  << sampled << ", margin " << margin << ")\n"
                  << std::flush;
    }

    const std::string kv_name =
        options.kv == ninfer::KvCacheStorage::Int8Group64 ? "int8" : "fp16";
    Json report{
        {"format", "ninfer_engine_top_logits_v1"},
        {"weights", std::filesystem::absolute(options.weights).string()},
        {"input_ids_file", std::filesystem::absolute(options.ids).string()},
        {"input_tokens", ids.size()},
        {"tensor_parallel_size", options.tp},
        {"devices", options.devices},
        {"prefill_chunk", options.prefill_chunk},
        {"kv_cache", kv_name},
        {"top_k", options.top_k},
        {"probes", std::move(probes)},
    };
    std::filesystem::create_directories(options.out.parent_path());
    std::ofstream output(options.out);
    if (!output) { throw std::runtime_error("cannot write report: " + options.out.string()); }
    output << report.dump(2) << "\n";
    std::cout << "engine_logits_probe: wrote " << options.out << "\n";
    return 0;
}
