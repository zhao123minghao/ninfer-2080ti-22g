// Implements: include/ninfer/ops/allreduce.h
//
// Host-side composition only: the transport is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over
// UVA pointers (see pull_peer() below -- deliberately NOT cudaMemcpyPeerAsync, which stream
// capture rejects), and the local combine reuses the qualified residual_add computation body
// (x += y in BF16 with FP32 accumulation and a single round-to-nearest-even on store), which is
// exactly this Op's local step. Sharing that private launch body keeps one implementation of the
// BF16 sum instead of a second, separately qualified copy of the same arithmetic.
//
// Both collectives share one three-phase issue order. The phases exist because a wait must not be
// issued before the record it observes: cudaStreamWaitEvent snapshots the event's current state,
// so phase B's wait on inputs_ready[1-r] would snapshot a stale (or absent) capture point if the
// peer's phase-A record had not been issued yet.
//
//   phase A, both ranks:  record(inputs_ready[r])
//   phase B, both ranks:  wait(inputs_ready[1-r]); pull peer source into own storage;
//                         record(pull_done[r])
//   phase C, both ranks:  wait(pull_done[1-r]); local combine (allreduce_sum only)
//
// THE PULL ITSELF is cudaMemcpyAsync with cudaMemcpyDeviceToDevice over UVA pointers, NOT
// cudaMemcpyPeerAsync -- see pull_peer() below for why. The choreography, the streams each call
// is issued on, and the ordering proof are unchanged by that choice: it is the same transfer
// expressed through the API that CUDA graph capture accepts.
#include "ninfer/ops/allreduce.h"

#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

void require_two_devices(const ExecutionContext& ec, const char* message) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value(), message);
    require(ec.dev[0]->device != ec.dev[1]->device, message);
}

std::uint8_t* byte_offset(void* base, std::size_t offset) {
    return static_cast<std::uint8_t*>(base) + offset;
}

// The inbound half of a pull: `bytes` from `source` (resident on the peer device) into
// `destination` (resident on the device `stream` belongs to), issued on the DESTINATION's stream.
//
// Deliberately NOT cudaMemcpyPeerAsync. That entry point is rejected inside a stream capture
// region with cudaErrorStreamCaptureUnsupported (measured on CUDA 13.1 / driver 580.178.04, Task
// 4.2's capture probe), which would make the entire tensor-parallel decode program uncapturable
// and cost the ~40-per-layer host launch overhead that CUDA Graphs exist to remove. Under unified
// virtual addressing -- which every 64-bit Linux CUDA context has -- a device pointer already
// names its device, so cudaMemcpyAsync with cudaMemcpyDeviceToDevice expresses exactly the same
// cross-device transfer: direct over PCIe when the driver granted peer access, transparently
// staged through host memory when it did not (GeForce-class boards), identical either way in
// bytes moved and stream ordering. Verified equal to the peer form both eagerly (this file's
// qualification suite) and under capture.
cudaError_t pull_peer(void* destination, const void* source, std::size_t bytes,
                      cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
}

// Current-device save/restore. Both collectives issue work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceGuard {
public:
    CurrentDeviceGuard() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceGuard() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceGuard(const CurrentDeviceGuard&)            = delete;
    CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void startup_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("peer transport startup: ") + operation + ": " +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
}

// Diagnostic context for a refused or failed direct peer route: which cards sit behind a
// translated IOMMU domain. Recorded in the startup message only. It is deliberately NOT a veto --
// see enable_peer_access() for the measurement that retired that shortcut.
std::string translated_iommu_domain(const ExecutionContext& ec) {
    std::string reason;
#if defined(__linux__)
    for (int rank = 0; rank < 2; ++rank) {
        char pci_bus_id[32]{};
        startup_check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), ec.dev[rank]->device),
                      "cudaDeviceGetPCIBusId");
        // CUDA may emit uppercase hexadecimal; Linux PCI sysfs names are lowercase.
        for (char& c : pci_bus_id) {
            if (c >= 'A' && c <= 'F') { c += 'a' - 'A'; }
        }
        std::ifstream domain_file(std::string("/sys/bus/pci/devices/") + pci_bus_id +
                                  "/iommu_group/type");
        std::string domain;
        domain_file >> domain;
        if (domain == "DMA" || domain == "DMA-FQ") {
            if (!reason.empty()) { reason += "; "; }
            reason += std::string("PCI ") + pci_bus_id + " uses translated IOMMU domain " + domain;
        }
    }
#else
    (void)ec;
#endif
    return reason;
}

// Startup-only storage. The exact payload catches drivers that advertise peer access
// but silently drop DMA writes (observed with V100s behind an IOMMU). Use the same
// pull API and destination compute stream as the collectives, without GPU peer loads.
class PeerTransferProbe {
public:
    explicit PeerTransferProbe(const ExecutionContext& ec) : ec_(ec) {
        startup_check(cudaGetDevice(&previous_), "cudaGetDevice");
    }

    ~PeerTransferProbe() {
        for (int rank = 0; rank < 2; ++rank) {
            cleanup(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
            if (source_[rank] != nullptr) { cleanup(cudaFree(source_[rank]), "cudaFree source"); }
            if (destination_[rank] != nullptr) {
                cleanup(cudaFree(destination_[rank]), "cudaFree destination");
            }
        }
        cleanup(cudaSetDevice(previous_), "restore device");
    }

    PeerTransferProbe(const PeerTransferProbe&) = delete;
    PeerTransferProbe& operator=(const PeerTransferProbe&) = delete;

    void set_device(int rank) const {
        startup_check(cudaSetDevice(ec_.dev[rank]->device), "cudaSetDevice");
    }

    void initialize(std::size_t payload_bytes) {
        payload_bytes_ = payload_bytes;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            startup_check(cudaMalloc(&source_[rank], payload_bytes_), "cudaMalloc source");
            startup_check(cudaMalloc(&destination_[rank], payload_bytes_),
                          "cudaMalloc destination");
            std::vector<std::uint32_t> values(payload_bytes_ / sizeof(std::uint32_t));
            for (std::size_t i = 0; i < values.size(); ++i) { values[i] = pattern(rank, i); }
            startup_check(cudaMemcpyAsync(source_[rank], values.data(), payload_bytes_,
                                           cudaMemcpyHostToDevice, ec_.dev[rank]->stream),
                          "initialize source");
            startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream), "retire source");
        }
    }

    // Empty means every probed size, both directions, and the whole-buffer sliced walk returned
    // the peer's exact payload.
    std::string qualify() {
        const std::size_t words = payload_bytes_ / sizeof(std::uint32_t);
        std::vector<std::uint32_t> actual(words);
        // Bracket the payload range rather than testing one convenient block. The reason a
        // domain-based veto looked attractive was the suspicion that a copy can pass at one size
        // while silently losing writes at another, so size is exactly what this has to vary.
        for (const std::size_t size : probe_sizes()) {
            const std::size_t size_words = size / sizeof(std::uint32_t);
            for (int rank = 0; rank < 2; ++rank) {
                set_device(rank);
                cudaStream_t stream = ec_.dev[rank]->stream;
                startup_check(cudaMemsetAsync(destination_[rank], 0xcd, size, stream),
                              "clear destination");
                startup_check(pull_peer(destination_[rank], source_[1 - rank], size, stream),
                              "cross-device copy");
                startup_check(cudaStreamSynchronize(stream), "retire cross-device copy");
                startup_check(cudaMemcpy(actual.data(), destination_[rank], size,
                                         cudaMemcpyDeviceToHost),
                              "read destination");
                for (std::size_t i = 0; i < size_words; ++i) {
                    if (actual[i] != pattern(1 - rank, i)) {
                        return describe(size, 1 - rank, rank, i);
                    }
                }
            }
        }
        // Then cover every mapping of the whole buffer with many separate small transfers: that is
        // the shape a decode round actually issues (128 independent 10 KiB collectives into
        // distinct staging regions), and the shape a single large copy cannot speak for.
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            cudaStream_t stream = ec_.dev[rank]->stream;
            startup_check(cudaMemsetAsync(destination_[rank], 0xcd, payload_bytes_, stream),
                          "clear destination");
            for (std::size_t offset = 0; offset < payload_bytes_; offset += kSliceBytes) {
                const std::size_t slice =
                    std::min(kSliceBytes, payload_bytes_ - offset);
                startup_check(pull_peer(byte_offset(destination_[rank], offset),
                                        byte_offset(source_[1 - rank], offset), slice, stream),
                              "sliced cross-device copy");
            }
            startup_check(cudaStreamSynchronize(stream), "retire sliced copies");
            startup_check(cudaMemcpy(actual.data(), destination_[rank], payload_bytes_,
                                     cudaMemcpyDeviceToHost),
                          "read destination");
            for (std::size_t i = 0; i < words; ++i) {
                if (actual[i] != pattern(1 - rank, i)) {
                    return describe(kSliceBytes, 1 - rank, rank, i);
                }
            }
        }
        return std::string();
    }

    // Validate the explicit fallback used by the collectives.  This is intentionally separate
    // from qualify(): the UVA D2D form exercises the driver's opaque staging path, while this
    // route must prove the two concrete D2H/H2D copies and their byte identity.
    std::string qualify_host_staged() {
        const std::size_t words = payload_bytes_ / sizeof(std::uint32_t);
        std::array<void*, 2> host{nullptr, nullptr};
        // Heap, not a stack array: the payload bound is the engine's real prefill collective size
        // (tens of MiB), so a std::array<std::uint32_t, words> here overruns the 8 MiB thread
        // stack. Only the pairs whose direct probe fails ever reach this route, so the overflow
        // stayed hidden on the NVLink pair. See history.md 5.4.
        std::vector<std::uint32_t> actual(words);
        std::string mismatch;
        try {
            for (void*& slot : host) {
                startup_check(cudaHostAlloc(&slot, payload_bytes_, cudaHostAllocPortable),
                              "cudaHostAlloc probe");
            }
            for (int rank = 0; rank < 2; ++rank) {
                set_device(rank);
                startup_check(cudaMemcpyAsync(host[rank], source_[rank], payload_bytes_,
                                              cudaMemcpyDeviceToHost, ec_.dev[rank]->stream),
                              "probe device-to-host");
                startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream),
                              "retire probe device-to-host");
            }
            for (int rank = 0; rank < 2; ++rank) {
                set_device(rank);
                startup_check(cudaMemcpyAsync(destination_[rank], host[1 - rank], payload_bytes_,
                                              cudaMemcpyHostToDevice, ec_.dev[rank]->stream),
                              "probe host-to-device");
                startup_check(cudaStreamSynchronize(ec_.dev[rank]->stream),
                              "retire probe host-to-device");
                startup_check(cudaMemcpy(actual.data(), destination_[rank], payload_bytes_,
                                         cudaMemcpyDeviceToHost), "read staged destination");
                for (std::size_t i = 0; i < words; ++i) {
                    if (actual[i] != pattern(1 - rank, i)) {
                        mismatch = "device " + std::to_string(ec_.dev[1 - rank]->device) +
                                   " -> " + std::to_string(ec_.dev[rank]->device) +
                                   " staged data mismatch at word " + std::to_string(i);
                        break;
                    }
                }
                if (!mismatch.empty()) { break; }
            }
        } catch (...) {
            for (void*& slot : host) {
                if (slot != nullptr) { (void)cudaFreeHost(slot); }
            }
            throw;
        }
        for (void*& slot : host) {
            if (slot != nullptr) { startup_check(cudaFreeHost(slot), "cudaFreeHost probe"); }
        }
        return mismatch;
    }

    void disable_peer_access() const {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            const cudaError_t status = cudaDeviceDisablePeerAccess(ec_.dev[1 - rank]->device);
            if (status == cudaErrorPeerAccessNotEnabled) {
                (void)cudaGetLastError();
            } else {
                startup_check(status, "cudaDeviceDisablePeerAccess");
            }
        }
    }

private:
    // The probe must bracket the payloads the collectives move rather than test one small block,
    // and its largest probed copy must be the real worst case: `payload_bytes_` is the declared
    // bound the host-staging slots are sized for, so the probe speaks for exactly the transfers
    // the collectives will issue. Peak cost is two `payload_bytes_` device buffers per rank,
    // allocated and released inside enable_peer_access() before any inference work is enqueued.
    static constexpr std::size_t kSliceBytes = 64U * 1024U;

    // Ascending, de-duplicated bracket of the sizes below the declared bound.
    [[nodiscard]] std::vector<std::size_t> probe_sizes() const {
        const std::array<std::size_t, 4> bracket{4u << 10, kSliceBytes, 1u << 20, 8u << 20};
        std::vector<std::size_t> sizes;
        for (const std::size_t candidate : bracket) {
            if (candidate < payload_bytes_) { sizes.push_back(candidate); }
        }
        sizes.push_back(payload_bytes_);
        return sizes;
    }

    std::string describe(std::size_t size, int source_device, int destination_device,
                         std::size_t word) const {
        return "device " + std::to_string(source_device) + " -> " +
               std::to_string(destination_device) + " copy of " + std::to_string(size) +
               " bytes mismatched at word " + std::to_string(word);
    }

    static std::uint32_t pattern(int rank, std::size_t index) {
        return 0x4f000000U ^ (std::uint32_t(rank) << 20U) ^
               (static_cast<std::uint32_t>(index) * 65537U);
    }

    static void cleanup(cudaError_t status, const char* operation) noexcept {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during peer probe %s: %s: %s\n",
                         operation, cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    const ExecutionContext& ec_;
    int previous_ = 0;
    std::size_t payload_bytes_ = 0;
    std::array<void*, 2> source_{};
    std::array<void*, 2> destination_{};
};

#ifndef NDEBUG
// Debug-only residency and aliasing predicates. These cost a driver round trip per pointer, so
// they are compiled out of the Release build the product ships; a wrong-device or self-overlapping
// argument is a caller bug that surfaces here during development instead of as a silently wrong
// result or an opaque cudaErrorInvalidValue later.
void require_resident_on(const void* pointer, int device, const char* message) {
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device, message);
}

void require_disjoint(const void* first, std::size_t first_bytes, const void* second,
                      std::size_t second_bytes, const char* message) {
    const auto* a = static_cast<const std::uint8_t*>(first);
    const auto* b = static_cast<const std::uint8_t*>(second);
    require(a + first_bytes <= b || b + second_bytes <= a, message);
}
#endif

} // namespace

bool enable_peer_access(const ExecutionContext& ec, std::size_t host_staging_bytes) {
    ec.direct_peer_access = false;
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    if (pair[0] == pair[1]) { return false; }
    require(host_staging_bytes > 0, "enable_peer_access: the declared payload bound must be > 0");

    PeerTransferProbe probe(ec);
    try {
        // The IOMMU domain type is recorded as context, never used as a decision. On this host
        // both cards report a translated DMA-FQ domain, yet cudaDeviceEnablePeerAccess succeeds
        // and every probed payload copies exactly, including a full staging-cap block up to 64 MiB
        // and a sliced walk over the whole buffer -- so reading the domain string first selected
        // the two-hop host-staged route on a machine whose peer link was fully usable, at the cost
        // of a bus round trip per collective. The failure that motivated the shortcut, a small
        // copy passing while other mappings lose writes, is a property to MEASURE; qualify()
        // measures it across the payload range and across every mapping.
        const std::string iommu_note = translated_iommu_domain(ec);

        int forward = 0;
        int reverse = 0;
        startup_check(cudaDeviceCanAccessPeer(&forward, pair[0], pair[1]),
                      "cudaDeviceCanAccessPeer forward");
        startup_check(cudaDeviceCanAccessPeer(&reverse, pair[1], pair[0]),
                      "cudaDeviceCanAccessPeer reverse");
        bool supported = forward != 0 && reverse != 0;
        // Diagnostic: force the two-hop host-staged route even on a pair whose peer link works. The
        // route is the only thing this changes, so the same binary can A/B it (set the variable for
        // one side of the pair of runs and not the other). Nothing else about the collective set
        // changes -- the same bytes land in the same staging buffer.
        const bool force_host_staged = std::getenv("NINFER_FORCE_HOST_STAGED") != nullptr;
        if (force_host_staged) { supported = false; }

        std::string direct_failure;
        if (supported) {
            for (int rank = 0; rank < 2; ++rank) {
                probe.set_device(rank);
                const cudaError_t status = cudaDeviceEnablePeerAccess(pair[1 - rank], 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled) {
                    (void)cudaGetLastError();
                } else {
                    startup_check(status, "cudaDeviceEnablePeerAccess");
                }
            }
        } else {
            direct_failure = force_host_staged ? "NINFER_FORCE_HOST_STAGED is set (diagnostic A/B)"
                                               : "peer access unavailable";
            probe.disable_peer_access();
        }
        probe.initialize(host_staging_bytes);
        if (supported) {
            direct_failure = probe.qualify();
            if (direct_failure.empty()) {
                ec.direct_peer_access = true;
                // State the route on success as well. The failure branch below is unconditional, so
                // a silent success was previously only inferable from a missing line -- which is
                // exactly the kind of claim that went unchecked while this function was refusing a
                // working link from the IOMMU domain string.
                std::fprintf(stderr,
                             "[ninfer] direct P2P enabled (every probed payload and mapping exact); "
                             "using direct device-to-device copies\n");
                return true;
            }
            probe.disable_peer_access();
        }

        const std::string staged_failure = probe.qualify_host_staged();
        if (!staged_failure.empty()) {
            throw std::runtime_error("peer transport startup: host-staged validation failed: " +
                                     staged_failure);
        }
        if (!iommu_note.empty()) { direct_failure += "; " + iommu_note; }
        std::fprintf(stderr,
                     "[ninfer] direct P2P disabled (%s); using verified pinned host-staged copies\n",
                     direct_failure.c_str());
        return false;
    } catch (...) {
        // Failed startup must not leave a partially enabled pair behind. Clear the
        // runtime's last error before cleanup; a fatal context error still prevents
        // further use, and the original startup exception remains authoritative.
        (void)cudaGetLastError();
        try { probe.disable_peer_access(); } catch (...) {}
        throw;
    }
}

PeerEvents::PeerEvents(const ExecutionContext& ec, std::size_t host_staging_bytes) {
    require_two_devices(ec, "PeerEvents: requires an ExecutionContext with two distinct devices");
    const CurrentDeviceGuard guard;
    // Create through a local table so a mid-way failure destroys what was already created instead
    // of leaking it; only a fully constructed set is published into the members.
    direct_transport_ = ec.direct_peer_access;
    for (int rank = 0; rank < 2; ++rank) {
        devices_[static_cast<std::size_t>(rank)] = ec.dev[rank]->device;
        streams_[static_cast<std::size_t>(rank)] = ec.dev[rank]->stream;
    }
    cudaEvent_t created[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int slot = 0; slot < 6; ++slot) {
        const int rank             = slot % 2;
        const cudaError_t creation = cudaSetDevice(ec.dev[rank]->device);
        cudaError_t status         = creation;
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&created[slot], cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            for (int done = 0; done < slot; ++done) { cudaEventDestroy(created[done]); }
            throw std::runtime_error(std::string("PeerEvents: event creation failed: ") +
                                     cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
        }
    }
    inputs_ready_ = {created[0], created[1]};
    transfer_ready_ = {created[2], created[3]};
    pull_done_      = {created[4], created[5]};
    // The pinned staging slots are allocated here, once, for the declared payload bound. They are
    // never moved afterwards: CUDA Graph capture bakes the host address into the staged route's
    // D2H/H2D copy nodes, so a later reallocation would leave every replayed collective reading
    // freed host memory. A collective that presents a larger payload reports an error instead --
    // see require_host_staging(). The bound has to cover every collective the program issues,
    // because a captured round may reach the largest prefill all-reduce before any eager call
    // could have sized the buffer (the failure history.md 5.4 records: a 64 MiB default against
    // an 8192-token prefill all-reduce of 80 MiB).
    try {
        require(host_staging_bytes > 0, "PeerEvents: the declared payload bound must be > 0");
        allocate_host_staging(host_staging_bytes);
    } catch (...) {
        for (cudaEvent_t& event : created) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
        for (void*& slot : host_staging_) {
            if (slot != nullptr) { (void)cudaFreeHost(slot); slot = nullptr; }
        }
        throw;
    }
    // A host slot is also read by the peer's H2D engine. Seed pull_done with a completed event so
    // the first staged collective may use the same inter-call lifetime edge as every later one.
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaEventRecord(pull_done_[static_cast<std::size_t>(rank)],
                                   ec.dev[rank]->stream));
    }
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceGuard::set(ec.dev[rank]->device);
        CUDA_CHECK(cudaStreamSynchronize(ec.dev[rank]->stream));
    }
    host_staging_ready_ = true;
}

PeerEvents::~PeerEvents() {
    for (std::array<cudaEvent_t, 2>* group : {&inputs_ready_, &transfer_ready_, &pull_done_}) {
        for (cudaEvent_t& event : *group) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaEventDestroy: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            event = nullptr;
        }
    }
    for (void*& slot : host_staging_) {
        if (slot != nullptr) {
            for (cudaStream_t stream : streams_) {
                const cudaError_t sync = cudaStreamSynchronize(stream);
                if (sync != cudaSuccess) {
                    std::fprintf(stderr, "CUDA cleanup failed during stream synchronization: %s: %s\n",
                                 cudaGetErrorName(sync), cudaGetErrorString(sync));
                }
            }
            const cudaError_t status = cudaFreeHost(slot);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "CUDA cleanup failed during cudaFreeHost: %s: %s\n",
                             cudaGetErrorName(status), cudaGetErrorString(status));
            }
            slot = nullptr;
        }
    }
}

PeerEvents::PeerEvents(PeerEvents&& other) noexcept
    : inputs_ready_(other.inputs_ready_), transfer_ready_(other.transfer_ready_),
      pull_done_(other.pull_done_), direct_transport_(other.direct_transport_),
      devices_(other.devices_), streams_(other.streams_),
      host_staging_(other.host_staging_), host_staging_bytes_(other.host_staging_bytes_),
      host_staging_ready_(other.host_staging_ready_) {
    other.inputs_ready_ = {nullptr, nullptr};
    other.transfer_ready_ = {nullptr, nullptr};
    other.pull_done_    = {nullptr, nullptr};
    other.devices_      = {0, 0};
    other.streams_      = {nullptr, nullptr};
    other.host_staging_ = {nullptr, nullptr};
    other.host_staging_bytes_ = 0;
    other.host_staging_ready_ = false;
}

PeerEvents& PeerEvents::operator=(PeerEvents&& other) noexcept {
    // Swap rather than destroy-then-assign: `other`'s destructor releases whatever this instance
    // held, in exactly one place.
    inputs_ready_.swap(other.inputs_ready_);
    transfer_ready_.swap(other.transfer_ready_);
    pull_done_.swap(other.pull_done_);
    std::swap(direct_transport_, other.direct_transport_);
    devices_.swap(other.devices_);
    streams_.swap(other.streams_);
    host_staging_.swap(other.host_staging_);
    std::swap(host_staging_bytes_, other.host_staging_bytes_);
    std::swap(host_staging_ready_, other.host_staging_ready_);
    return *this;
}

void PeerEvents::require_host_staging(std::size_t bytes) const {
    if (bytes <= host_staging_bytes_) { return; }
    throw std::length_error(
        "host-staged collective payload of " + std::to_string(bytes) +
        " bytes exceeds the pinned staging allocated for " + std::to_string(host_staging_bytes_) +
        " bytes; the staging cannot be relocated once a CUDA Graph has captured its address, so "
        "the peer events must be constructed with a bound that covers every collective");
}

void PeerEvents::allocate_host_staging(std::size_t bytes) const {
    require(bytes > 0 && host_staging_bytes_ == 0 && !host_staging_ready_,
            "PeerEvents: host staging is allocated exactly once");
    try {
        for (void*& slot : host_staging_) {
            CUDA_CHECK(cudaHostAlloc(&slot, bytes, cudaHostAllocPortable));
        }
        host_staging_bytes_ = bytes;
        host_staging_ready_ = true;
    } catch (...) {
        for (void*& slot : host_staging_) {
            if (slot != nullptr) { (void)cudaFreeHost(slot); slot = nullptr; }
        }
        host_staging_bytes_ = 0;
        throw;
    }
}

void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allreduce_sum: requires an ExecutionContext with two distinct devices");
    for (int rank = 0; rank < 2; ++rank) {
        require(buffer[rank].dtype == DType::BF16 && staging[rank].dtype == DType::BF16,
                "allreduce_sum: buffer/staging must be BF16");
        require(buffer[rank].data != nullptr && staging[rank].data != nullptr,
                "allreduce_sum: buffer/staging data must be non-null");
        require(buffer[rank].is_contiguous() && staging[rank].is_contiguous(),
                "allreduce_sum: buffer/staging must be contiguous");
        for (int d = 0; d < 4; ++d) {
            require(buffer[rank].ne[d] == buffer[0].ne[d] && staging[rank].ne[d] == buffer[0].ne[d],
                    "allreduce_sum: buffer/staging shapes must match on both devices");
        }
    }
    require(events.live(), "allreduce_sum: events must be live");

    const std::size_t bytes = buffer[0].bytes();
    if (bytes == 0) { return; }
    events.require_host_staging(bytes);

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(buffer[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: buffer[r] must be resident on ec.dev[r]");
        require_resident_on(staging[rank].data, ec.dev[rank]->device,
                            "allreduce_sum: staging[r] must be resident on ec.dev[r]");
        require_disjoint(buffer[rank].data, bytes, staging[rank].data, bytes,
                         "allreduce_sum: staging[r] must not overlap buffer[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my operand is complete" on each stream, before any wait observes it.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: direct P2P is one pull.  Behind translated IOMMU domains, make the two PCIe
    // transfers explicit: both D2H legs are issued first, then both ranks wait for the peer's
    // host slot and issue H2D.  The host slots are source-owned and the existing pull_done edge
    // still prevents a source buffer from being overwritten while its peer reads it.
    if (events.direct_transport()) {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
            CUDA_CHECK(pull_peer(staging[rank].data, buffer[1 - rank].data, bytes, local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    } else {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(events.host_staging(rank), buffer[rank].data, bytes,
                                       cudaMemcpyDeviceToHost, local.stream));
            CUDA_CHECK(cudaEventRecord(events.transfer_ready(rank), local.stream));
        }
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.transfer_ready(1 - rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(staging[rank].data, events.host_staging(1 - rank), bytes,
                                       cudaMemcpyHostToDevice, local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    }

    // Phase C: the in-place combine may only overwrite buffer[rank] once the peer has finished
    // reading it. That same wait is what makes the next call's phase B safe.
    //
    // The wait cannot be hoisted into phase B, ahead of the export whose slot it protects, even
    // though it would then be off this call's critical path: at that point the peer's stream has
    // not joined a stream capture yet -- the fork is established by the wait(transfer_ready) in
    // phase B's second loop -- so waiting there on the peer event's previous, uncaptured record
    // raises cudaErrorStreamCaptureIsolation. Measured, not inferred; see history.md section 38.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
        Tensor accumulator = buffer[rank];
        detail::residual_add_launch(staging[rank], accumulator, local.stream);
    }
}

void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec,
                        "allgather_rows: requires an ExecutionContext with two distinct devices");
    const DType dtype             = destination[0].dtype;
    const std::int32_t row_length = destination[0].ne[0];
    const std::int32_t total_rows = destination[0].ne[1];
    for (int rank = 0; rank < 2; ++rank) {
        require(destination[rank].dtype == dtype && part[rank].dtype == dtype,
                "allgather_rows: destination/part must share one dtype");
        require(destination[rank].data != nullptr && part[rank].data != nullptr,
                "allgather_rows: destination/part data must be non-null");
        require(destination[rank].is_contiguous() && part[rank].is_contiguous(),
                "allgather_rows: destination/part must be contiguous");
        require(destination[rank].ne[0] == row_length && part[rank].ne[0] == row_length,
                "allgather_rows: destination/part must agree on row length ne[0]");
        require(destination[rank].ne[1] == total_rows,
                "allgather_rows: both destinations must have the same row count");
        require(destination[rank].ne[2] == 1 && destination[rank].ne[3] == 1 &&
                    part[rank].ne[2] == 1 && part[rank].ne[3] == 1,
                "allgather_rows: destination/part must be two-dimensional [C, R]");
    }
    require(part[0].ne[1] + part[1].ne[1] == total_rows,
            "allgather_rows: owned row counts must sum to the destination row count");
    require(events.live(), "allgather_rows: events must be live");

    const std::size_t row_bytes = static_cast<std::size_t>(row_length) * dtype_size(dtype);
    const std::size_t block[2]  = {row_bytes * static_cast<std::size_t>(part[0].ne[1]),
                                   row_bytes * static_cast<std::size_t>(part[1].ne[1])};
    const std::size_t offset[2] = {0, block[0]};
    events.require_host_staging(std::max(block[0], block[1]));

#ifndef NDEBUG
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(destination[rank].data, ec.dev[rank]->device,
                            "allgather_rows: destination[r] must be resident on ec.dev[r]");
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "allgather_rows: part[r] must be resident on ec.dev[r]");
        require_disjoint(destination[rank].data, destination[rank].bytes(), part[rank].data,
                         block[rank], "allgather_rows: part[r] must not overlap destination[r]");
    }
#endif

    const CurrentDeviceGuard guard;

    // Phase A: publish "my block is complete".
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: each destination is written only by its own stream.  The staged branch first
    // exports each owned block to its source slot, then imports the peer slot after its event.
    if (events.direct_transport()) {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                       part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                       local.stream));
            CUDA_CHECK(pull_peer(byte_offset(destination[rank].data, offset[1 - rank]),
                                 part[1 - rank].data, block[1 - rank], local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    } else {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(events.host_staging(rank), part[rank].data, block[rank],
                                       cudaMemcpyDeviceToHost, local.stream));
            CUDA_CHECK(cudaEventRecord(events.transfer_ready(rank), local.stream));
            CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[rank]),
                                       part[rank].data, block[rank], cudaMemcpyDeviceToDevice,
                                       local.stream));
        }
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceContext& local = *ec.dev[rank];
            CurrentDeviceGuard::set(local.device);
            CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.transfer_ready(1 - rank), 0));
            CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination[rank].data, offset[1 - rank]),
                                       events.host_staging(1 - rank), block[1 - rank],
                                       cudaMemcpyHostToDevice, local.stream));
            CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
        }
    }

    // Phase C: the Op writes nothing else, but the caller (or the next call) will overwrite
    // part[rank]. Ordering each stream after the peer's read is what makes that safe without a
    // host synchronization. As in allreduce_sum, this wait cannot be hoisted ahead of the export:
    // the peer's stream has not joined a capture that early.
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CurrentDeviceGuard::set(local.device);
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
    }
}

void gather_columns_rank0(const Tensor& destination, const std::array<Tensor, 2>& part,
                          const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(
        ec, "gather_columns_rank0: requires an ExecutionContext with two distinct devices");
    const DType dtype = destination.dtype;
    const std::int32_t full_width = destination.ne[0];
    const std::int32_t columns = destination.ne[1];
    require(full_width > 0 && columns > 0 && destination.data != nullptr &&
                destination.is_contiguous() && destination.ne[2] == 1 && destination.ne[3] == 1,
            "gather_columns_rank0: destination must be contiguous [C,T]");
    for (int rank = 0; rank < 2; ++rank) {
        require(part[rank].dtype == dtype && part[rank].data != nullptr &&
                    part[rank].is_contiguous() && part[rank].ne[0] > 0 &&
                    part[rank].ne[1] == columns && part[rank].ne[2] == 1 &&
                    part[rank].ne[3] == 1,
                "gather_columns_rank0: parts must be contiguous [C_r,T] tensors of one dtype");
    }
    require(part[0].ne[0] + part[1].ne[0] == full_width,
            "gather_columns_rank0: shard widths must sum to destination width");
    require(events.live(), "gather_columns_rank0: events must be live");

    const std::size_t element_bytes = dtype_size(dtype);
    const std::size_t destination_pitch = static_cast<std::size_t>(full_width) * element_bytes;
    const std::size_t block[2] = {
        static_cast<std::size_t>(part[0].ne[0]) * element_bytes,
        static_cast<std::size_t>(part[1].ne[0]) * element_bytes};
    const std::size_t offset[2] = {0, block[0]};
    events.require_host_staging(block[1]);

#ifndef NDEBUG
    require_resident_on(destination.data, ec.dev[0]->device,
                        "gather_columns_rank0: destination must be resident on rank 0");
    for (int rank = 0; rank < 2; ++rank) {
        require_resident_on(part[rank].data, ec.dev[rank]->device,
                            "gather_columns_rank0: part[r] must be resident on its rank");
        require_disjoint(destination.data, destination.bytes(), part[rank].data,
                         part[rank].bytes(),
                         "gather_columns_rank0: parts must not overlap destination");
    }
#endif

    const CurrentDeviceGuard guard;
    const DeviceContext& rank0 = *ec.dev[0];
    const DeviceContext& rank1 = *ec.dev[1];
    CurrentDeviceGuard::set(rank0.device);
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), rank0.stream));
    CurrentDeviceGuard::set(rank1.device);
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(1), rank1.stream));

    CurrentDeviceGuard::set(rank0.device);
    CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.inputs_ready(0), 0));
    CUDA_CHECK(cudaMemcpy2DAsync(destination.data, destination_pitch, part[0].data, block[0],
                                 block[0], static_cast<std::size_t>(columns),
                                 cudaMemcpyDeviceToDevice, rank0.stream));
    if (events.direct_transport()) {
        CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.inputs_ready(1), 0));
        CUDA_CHECK(cudaMemcpy2DAsync(byte_offset(destination.data, offset[1]), destination_pitch,
                                     part[1].data, block[1], block[1],
                                     static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice,
                                     rank0.stream));
    } else {
        CurrentDeviceGuard::set(rank1.device);
        CUDA_CHECK(cudaStreamWaitEvent(rank1.stream, events.inputs_ready(1), 0));
        CUDA_CHECK(cudaMemcpyAsync(events.host_staging(1), part[1].data, part[1].bytes(),
                                   cudaMemcpyDeviceToHost, rank1.stream));
        CUDA_CHECK(cudaEventRecord(events.transfer_ready(1), rank1.stream));
        CurrentDeviceGuard::set(rank0.device);
        CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.transfer_ready(1), 0));
        CUDA_CHECK(cudaMemcpy2DAsync(byte_offset(destination.data, offset[1]), destination_pitch,
                                     events.host_staging(1), block[1], block[1],
                                     static_cast<std::size_t>(columns), cudaMemcpyHostToDevice,
                                     rank0.stream));
    }
    CUDA_CHECK(cudaEventRecord(events.pull_done(0), rank0.stream));

    // Rank 1 may reuse its source and pinned slot only after rank 0 has consumed them.
    CurrentDeviceGuard::set(rank1.device);
    CUDA_CHECK(cudaStreamWaitEvent(rank1.stream, events.pull_done(0), 0));
}

void broadcast_rank0(const Tensor& source, const Tensor& destination,
                     const ExecutionContext& ec, const PeerEvents& events) {
    require_two_devices(ec, "broadcast_rank0: requires two devices");
    require(events.live(), "broadcast_rank0: events must be live");
    require(source.data != nullptr && destination.data != nullptr &&
                source.dtype == destination.dtype && source.is_contiguous() &&
                destination.is_contiguous(), "broadcast_rank0: invalid tensors");
    for (int d = 0; d < 4; ++d) {
        require(source.ne[d] == destination.ne[d], "broadcast_rank0: shapes must match");
    }
    const std::size_t bytes = source.bytes();
    if (bytes == 0) { return; }
    events.require_host_staging(bytes);
#ifndef NDEBUG
    require_resident_on(source.data, ec.dev[0]->device, "broadcast_rank0: source is not on rank 0");
    require_resident_on(destination.data, ec.dev[1]->device,
                        "broadcast_rank0: destination is not on rank 1");
#endif
    const CurrentDeviceGuard guard;
    const DeviceContext& origin = *ec.dev[0];
    const DeviceContext& peer = *ec.dev[1];
    CurrentDeviceGuard::set(origin.device);
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), origin.stream));
    if (events.direct_transport()) {
        CurrentDeviceGuard::set(peer.device);
        CUDA_CHECK(cudaStreamWaitEvent(peer.stream, events.inputs_ready(0), 0));
        CUDA_CHECK(pull_peer(destination.data, source.data, bytes, peer.stream));
    } else {
        CurrentDeviceGuard::set(origin.device);
        CUDA_CHECK(cudaMemcpyAsync(events.host_staging(0), source.data, bytes,
                                   cudaMemcpyDeviceToHost, origin.stream));
        CUDA_CHECK(cudaEventRecord(events.transfer_ready(0), origin.stream));
        CurrentDeviceGuard::set(peer.device);
        CUDA_CHECK(cudaStreamWaitEvent(peer.stream, events.transfer_ready(0), 0));
        CUDA_CHECK(cudaMemcpyAsync(destination.data, events.host_staging(0), bytes,
                                   cudaMemcpyHostToDevice, peer.stream));
    }
    CUDA_CHECK(cudaEventRecord(events.pull_done(1), peer.stream));
    CurrentDeviceGuard::set(origin.device);
    CUDA_CHECK(cudaStreamWaitEvent(origin.stream, events.pull_done(1), 0));
}

} // namespace ninfer::ops
