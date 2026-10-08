#include "ops/linear/fp8_block/fp8_block.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/launcher/kernel_attr_once.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlockRows = 128;
constexpr int kThreads = 128;
constexpr int kRowsPerBlock = 4;
constexpr int kTokensPerBlock = 4;

constexpr int kHmmaBlockRows = 128;
constexpr int kHmmaBlockCols = 128;
constexpr int kHmmaBlockK = 64;
constexpr int kHmmaDefaultStages = 1;
constexpr int kHmmaWarpRows = 64;
constexpr int kHmmaWarpCols = 32;
constexpr int kHmmaWarps = (kHmmaBlockRows / kHmmaWarpRows) * (kHmmaBlockCols / kHmmaWarpCols);
constexpr int kHmmaThreads = kHmmaWarps * 32;
constexpr int kHmmaRows = kHmmaWarpRows / 16;
constexpr int kHmmaCols = kHmmaWarpCols / 8;
constexpr int kHmmaKSub = kHmmaBlockK / 16;
constexpr int kHmmaItems = kHmmaBlockRows * (kHmmaBlockK / 8) / kHmmaThreads;
template <int Stages>
constexpr int hmma_shared_bytes() {
    return Stages * kHmmaBlockRows * kHmmaBlockK * static_cast<int>(sizeof(__half)) +
           Stages * kHmmaBlockCols * kHmmaBlockK * static_cast<int>(sizeof(__half));
}

enum class Fp8BlockRoute { Auto, Scalar, Hmma };
enum class Fp8BlockOperation { Linear, LinearAdd, SwiGlu, Attention, Gdn };

struct Fp8BlockAutoRoute {
    Fp8BlockOperation operation;
    int rows;
    int columns;
    int hmma_min_tokens;
};

constexpr int kAnyDimension = 0;
// Scalar/HMMA crossover, re-measured after the scalar kernel's activation loads were widened. The
// scalar kernel runs four tokens per CTA (`kTokensPerBlock`), so its cost steps with
// ceil(T / 4) blocks while the HMMA tile is flat in T until it is full; the crossover is
// therefore a block-count boundary, not a smooth crossing. Measured at T = 4/5/6/8/12 on this
// target: the wide-K rows cross at the fourth block, everything else at the third.
// Diagnostic-only compile-time phase ablation for the fused SwiGLU leaf. 0 keeps the production
// kernel; configuring the build with -DNINFER_FP8_BLOCK_ABLATION=1 removes the mma, 2 removes the
// decode/STS staging, and 3 removes the shared-memory fragment reads. Ablated builds are timing
// instruments only and produce wrong results.
#ifndef NINFER_FP8_BLOCK_ABLATION
#define NINFER_FP8_BLOCK_ABLATION 0
#endif

// Diagnostic-only compile-time ablation for the row-major scalar decode leaf. 0 keeps the
// production leaf; 1 replaces the two shared-table gathers of each code word with a linear code
// reinterpretation, removing the LDS hop from the LDG -> LDS -> FMA chain so that hop's cost can be
// measured on its own. Ablated builds are timing instruments only and produce wrong results.
#ifndef NINFER_FP8_SCALAR_ABLATION
#define NINFER_FP8_SCALAR_ABLATION 0
#endif

#if NINFER_FP8_SCALAR_ABLATION == 1
// Timing-only stand-in for decode_fp8_pair: same shape, no shared memory, no table.
__device__ __forceinline__ float2 decode_fp8_pair_ablated(std::uint32_t packed) {
    return make_float2(static_cast<float>(packed & 0xFFu),
                       static_cast<float>((packed >> 8) & 0xFFu));
}
#endif

constexpr Fp8BlockAutoRoute kFp8BlockAutoRoutes[] = {
    {Fp8BlockOperation::Linear, 5120, 17408, 13},
    {Fp8BlockOperation::Linear, kAnyDimension, kAnyDimension, 9},
    {Fp8BlockOperation::LinearAdd, kAnyDimension, kAnyDimension, 9},
    {Fp8BlockOperation::SwiGlu, kAnyDimension, kAnyDimension, 9},
    {Fp8BlockOperation::Attention, kAnyDimension, kAnyDimension, 9},
    {Fp8BlockOperation::Gdn, kAnyDimension, kAnyDimension, 13},
};

Fp8BlockRoute fp8_block_route() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_ROUTE");
    if (value == nullptr || std::strcmp(value, "auto") == 0) { return Fp8BlockRoute::Auto; }
    if (std::strcmp(value, "scalar") == 0) { return Fp8BlockRoute::Scalar; }
    if (std::strcmp(value, "hmma") == 0) { return Fp8BlockRoute::Hmma; }
    throw std::invalid_argument(
        "NINFER_FP8_BLOCK_ROUTE must be auto, scalar, or hmma");
}

bool fp8_block_persistent() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_PERSISTENT");
    if (value == nullptr) { return false; }
    if (std::strcmp(value, "0") == 0) { return false; }
    if (std::strcmp(value, "1") == 0) { return true; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_PERSISTENT must be 0 or 1");
}

int fp8_block_hmma_stages() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_HMMA_STAGES");
    if (value == nullptr) { return kHmmaDefaultStages; }
    if (std::strcmp(value, "1") == 0) { return 1; }
    if (std::strcmp(value, "2") == 0) { return 2; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_HMMA_STAGES must be 1 or 2");
}

bool fp8_block_hmma_scale_broadcast() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_HMMA_SCALE_BROADCAST");
    if (value == nullptr || std::strcmp(value, "1") == 0) { return true; }
    if (std::strcmp(value, "0") == 0) { return false; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_HMMA_SCALE_BROADCAST must be 0 or 1");
}

// Accumulate the prefill HMMA leaves entirely in fp16 (no per-K-tile fold into fp32), matching
// the external Marlin kernel's sm_75 `use_fp16_accum` path. This drops the 160-instruction fold
// from the K-loop body (27% of the 590-instruction loop) at the cost of fp16 accumulation over
// the full reduction. The external Marlin kernel ships this numerics and the source-aligned
// quality gate already validates fp16 accumulation, so this is the same arithmetic profile, not a
// new precision regime. Measured: swiglu leaf 44.9 -> 54.7 TFLOPS (the fold path also spills 104 B
// of stack; this path has STACK 0), engine 32K/512 prefill +10%, and the three-workload source
// gate stays at 98/99 argmax with the same single FP16/BF16 tie flip. 0 restores the fp16-partial
// + fold-to-fp32 path for A/B.
bool fp8_block_nofold() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_NOFOLD");
    if (value == nullptr || std::strcmp(value, "1") == 0) { return true; }
    if (std::strcmp(value, "0") == 0) { return false; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_NOFOLD must be 0 or 1");
}

// Prototype switch for the lookup-free ALU weight decode in the staged prefill HMMA kernels.
// 1 decodes each staged fp8 pair with decode_fp8_pair_alu and folds the 2^8 exponent bias into
// the BF16 scale (fp8_block_stage_scale_bias8); 0 keeps the 256-entry shared-table decode.
// Measured (same-session alternating): linear leaf 41.17 -> 46.00 TFLOPS (+11.7%), swiglu leaf
// +1.2%, engine 32K/512 prefill 1189.9 -> 1264.5 t/s (+6.3%), decode and spec acc bit-identical,
// oracle max_abs_error bit-identical. The LUT's 8 LDS + 4 PRMT per staged word contend with the
// staging stores and ldmatrix on sm_75's single LSU; the ALU route moves that to the ALU pipe.
bool fp8_block_alu_decode() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_ALU_DECODE");
    if (value == nullptr || std::strcmp(value, "1") == 0) { return true; }
    if (std::strcmp(value, "0") == 0) { return false; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_ALU_DECODE must be 0 or 1");
}

// Prototype switch for the register-dequantized SwiGLU HMMA kernel.
bool fp8_block_regdequant() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_REGDEQUANT");
    if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
    if (std::strcmp(value, "1") == 0) { return true; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_REGDEQUANT must be 0 or 1");
}

// Prototype switch for the packed 16x32-layout register-dequantized SwiGLU HMMA kernel.
bool fp8_block_packed() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_PACKED");
    if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
    if (std::strcmp(value, "1") == 0) { return true; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_PACKED must be 0 or 1");
}

// Prototype switch for the shared-broadcast register-dequantized SwiGLU HMMA kernel.
bool fp8_block_broadcast() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_BROADCAST");
    if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
    if (std::strcmp(value, "1") == 0) { return true; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_BROADCAST must be 0 or 1");
}

// Prototype switch for the small-tile (64x64) register-dequantized SwiGLU HMMA kernel.
bool fp8_block_smalltile() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_SMALLTILE");
    if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
    if (std::strcmp(value, "1") == 0) { return true; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_SMALLTILE must be 0 or 1");
}

// Prototype switch for the shared-broadcast register-dequantized linear HMMA kernel (milestone-2
// scheme C applied to the single-GEMM linear leaf: raw codes staged once per block, per-warp
// register decode straight into mma A fragments).
bool fp8_block_linear_broadcast() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_LINEAR_BROADCAST");
    if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
    if (std::strcmp(value, "1") == 0) { return true; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_LINEAR_BROADCAST must be 0 or 1");
}

// Prototype switch for the packed-layout (16x32 unit) register-dequantized linear HMMA kernel:
// the weight codes live in a persistent 16x32 layout so each lane's LDG.128 feeds two k-steps of
// mma with no shared staging and no weight ldmatrix. Routed by NINFER_FP8_BLOCK_LINEAR_PACKED.
bool fp8_block_linear_packed() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_LINEAR_PACKED");
    if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
    if (std::strcmp(value, "1") == 0) { return true; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_LINEAR_PACKED must be 0 or 1");
}

// Prototype switch for the 64-token x 256-N packed broadcast kernel (the external Marlin tile
// orientation: narrow token band, wide N band, 4 token x 2 N warp grid). Routed by
// NINFER_FP8_BLOCK_LINEAR_PACKED256.
bool fp8_block_linear_packed256() {
    const char* value = std::getenv("NINFER_FP8_BLOCK_LINEAR_PACKED256");
    if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
    if (std::strcmp(value, "1") == 0) { return true; }
    throw std::invalid_argument("NINFER_FP8_BLOCK_LINEAR_PACKED256 must be 0 or 1");
}

bool use_hmma(const Tensor& x, const Weight& weight, Fp8BlockOperation operation) {
    const Fp8BlockRoute route = fp8_block_route();
    if (route == Fp8BlockRoute::Hmma) { return true; }
    if (route == Fp8BlockRoute::Scalar) { return false; }
    for (const Fp8BlockAutoRoute& plan : kFp8BlockAutoRoutes) {
        if (plan.operation != operation ||
            (plan.rows != kAnyDimension && plan.rows != weight.n) ||
            (plan.columns != kAnyDimension && plan.columns != weight.k)) {
            continue;
        }
        return x.ne[1] >= plan.hmma_min_tokens;
    }
    throw std::invalid_argument("block-FP8 auto route is missing a supported operation");
}

static_assert(kHmmaItems * kHmmaThreads == kHmmaBlockRows * (kHmmaBlockK / 8));
static_assert(hmma_shared_bytes<1>() <= 64 * 1024);
static_assert(hmma_shared_bytes<2>() <= 64 * 1024);

struct LinearOutput {
    __nv_bfloat16* out;
    int rows;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        out[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(value);
    }
};

struct LinearAddOutput {
    __nv_bfloat16* residual;
    int rows;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const std::int64_t index = static_cast<std::int64_t>(token) * rows + row;
        residual[index] = __float2bfloat16_rn(value + __bfloat162float(residual[index]));
    }
};

struct AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* gate;
    __nv_bfloat16* key;
    __nv_bfloat16* value;
    int rows;

    __device__ __forceinline__ void store(int row, int token, float result) const {
        const int query_rows = rows == 14336 ? 6144 : 3072;
        const int key_value_rows = rows == 14336 ? 1024 : 512;
        if (row < query_rows) {
            query[static_cast<std::int64_t>(token) * query_rows + row] =
                __float2bfloat16_rn(result);
        } else if (row < query_rows + key_value_rows) {
            const int local = row - query_rows;
            key[static_cast<std::int64_t>(token) * key_value_rows + local] =
                __float2bfloat16_rn(result);
        } else if (row < 2 * query_rows + key_value_rows) {
            const int local = row - query_rows - key_value_rows;
            gate[static_cast<std::int64_t>(token) * query_rows + local] =
                __float2bfloat16_rn(result);
        } else {
            const int local = row - 2 * query_rows - key_value_rows;
            value[static_cast<std::int64_t>(token) * key_value_rows + local] =
                __float2bfloat16_rn(result);
        }
    }
};

struct GdnOutput {
    __nv_bfloat16* qkv;
    __nv_bfloat16* z;
    int rows;

    __device__ __forceinline__ void store(int row, int token, float result) const {
        const int qkv_rows = rows == 16384 ? 10240 : 5120;
        if (row < qkv_rows) {
            qkv[static_cast<std::int64_t>(token) * qkv_rows + row] =
                __float2bfloat16_rn(result);
        } else {
            const int z_rows = rows - qkv_rows;
            z[static_cast<std::int64_t>(token) * z_rows + row - qkv_rows] =
                __float2bfloat16_rn(result);
        }
    }
};

// On sm_75 `static_cast<__half2>(__nv_fp8x2_e4m3)` has no instruction behind it: ptxas emits a
// branchy software expansion (byte extraction, exponent classification, a subnormal
// normalization loop, then the fp16 conversion) costing on the order of fifteen instructions
// per two codes. Every staged weight word is re-decoded once per 128-token block, so that cost
// is multiplied by the token-block count and dominates the prefill kernels.
//
// The block-FP8 codes are arbitrary but the mapping code -> half is a fixed 256-entry function
// that is exact (E4M3 normals, subnormals and the 0x7f/0xff NaN payload are all representable
// in half). Materialise that function once per block from the reference conversion and read it
// as two 32-bit words whose halves are already duplicated, so a pair becomes two shared loads
// plus one byte permute.
constexpr int kFp8CodeCount = 256;

__device__ __forceinline__ void build_fp8_code_table(unsigned* table, int tid) {
    for (int code = tid; code < kFp8CodeCount; code += static_cast<int>(blockDim.x)) {
        __nv_fp8x2_e4m3 value;
        value.__x = static_cast<std::uint16_t>(code * 0x0101);
        const __half2 decoded = static_cast<__half2>(value);
        const unsigned half_bits = __half_as_ushort(__low2half(decoded));
        table[code] = half_bits | (half_bits << 16);
    }
}

__device__ __forceinline__ __half2 decode_fp8_pair_half(const unsigned* table,
                                                        std::uint16_t packed) {
    const unsigned low = table[packed & 0xFFu];
    const unsigned high = table[(packed >> 8) & 0xFFu];
    const unsigned bits = __byte_perm(low, high, 0x5410);
    return *reinterpret_cast<const __half2*>(&bits);
}

__device__ __forceinline__ void build_fp8_code_table_float(float* table, int tid,
                                                            int threads) {
    for (int code = tid; code < kFp8CodeCount; code += threads) {
        __nv_fp8x2_e4m3 value;
        value.__x = static_cast<std::uint16_t>(code * 0x0101);
        table[code] = static_cast<float2>(value).x;
    }
}

__device__ __forceinline__ float2 decode_fp8_pair(const float* table, std::uint16_t packed) {
    return make_float2(table[packed & 0xFFu], table[(packed >> 8) & 0xFFu]);
}

__device__ __forceinline__ std::uint32_t marlin_fp8_block_word(const std::uint8_t* codes,
                                                                int row, int column,
                                                                int output_rows) {
    const int tile_n = row / 32;
    const int tile_k = column / 32;
    const int local_n = row & 31;
    const int local_k = column & 31;
    const int thread = 4 * (local_n & 7) + ((local_k & 15) >> 2);
    const int warp = 2 * (local_n >> 4) + ((local_n >> 3) & 1);
    const int result = local_k >> 4;
    const int code_offset = (((tile_k * (output_rows / 32) + tile_n) * 32 + thread) * 4 + warp) * 8 +
                            result * 4;
    return *reinterpret_cast<const std::uint32_t*>(codes + code_offset);
}

// The compact scale plane is `[N/128][K/128]` BF16, the same plane the block-scale layout uses:
// one multiplier per 128x128 block, read once per row block and K group instead of once per staged
// element. Both Marlin leaves and the row-major leaves therefore index it the same way.
__device__ __forceinline__ int fp8_block_scale_index(int row, int k_group, int k_groups) {
    return (row >> 7) * k_groups + k_group;
}

// Four consecutive bf16 activations widen with a shift and a mask: a bf16 is the top half of the
// fp32 it widens to, so `bits << 16` and `bits & 0xFFFF0000` are exact and cost two ALU ops for
// both halves. Indexing the elements one at a time instead makes ptxas emit four LDG.U16, and with
// the lanes 8 bytes apart each of those fetches four times the bytes it uses.
__device__ __forceinline__ float2 bf16x2_wide(unsigned bits) {
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xFFFF0000u));
}

__device__ __forceinline__ int fp8_block_swizzle(int row, int column) {
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}


// One source word decodes to a single 16-byte row unit, and `fp8_block_swizzle` places every unit
// on one whole four-word bank group. The eight lanes of a group hold all eight groups, so the
// staged store must be 16 bytes wide: a four-byte store instruction can only ever reach the eight
// bank groups the eight lanes of a row produce, which serializes four ways. Staging the same bytes
// as four separate stores cost 1024 of the 1164 shared wavefronts the SwiGLU leaf spent per CTA K
// tile. Assembling the decoded halves first makes the store one 128-bit access at the hardware
// minimum for the 32 KiB weight tile.
__device__ __forceinline__ void stage_fp8_word(const unsigned* table, __half* destination,
                                               uint2 packed, __half2 scale) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
    unsigned words[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const std::uint16_t bits = static_cast<std::uint16_t>(bytes[2 * pair]) |
                                   (static_cast<std::uint16_t>(bytes[2 * pair + 1]) << 8);
        const __half2 values = __hmul2(decode_fp8_pair_half(table, bits), scale);
        words[pair] = *reinterpret_cast<const unsigned*>(&values);
    }
    store_vec(destination, make_uint4(words[0], words[1], words[2], words[3]));
}

// Register-side dequantization for the HMMA A fragment: two adjacent codes on one weight row
// decode and scale into the two f16 of one mma A register (low = code[k], high = code[k+1]),
// matching the bit order stage_fp8_word + ldmatrix produced. This skips shared staging entirely:
// the lane reads its two codes straight from the row-major packed weight plane.
__device__ __forceinline__ unsigned fp8_dequant_a_word(const unsigned* table,
                                                       const std::uint8_t* codes, int row, int k,
                                                       int input_rows, __half2 scale) {
    const std::uint8_t* source = codes + static_cast<std::int64_t>(row) * input_rows + k;
    const std::uint16_t packed = *reinterpret_cast<const std::uint16_t*>(source);
    const __half2 values = __hmul2(decode_fp8_pair_half(table, packed), scale);
    return *reinterpret_cast<const unsigned*>(&values);
}

// Register-side dequantization of one packed 16x32 layout unit: 16 consecutive codes decode into
// eight f16 pairs that are exactly two m16k8 A fragments' worth of registers (a0..a3 for k-step 0,
// a0..a3 for k-step 1). One LDG.128 feeds two k-steps with no shared staging.
__device__ __forceinline__ void fp8_dequant_a_unit(const unsigned* table, uint4 raw,
                                                   __half2 scale, unsigned (&out)[8]) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const std::uint16_t bits = static_cast<std::uint16_t>(bytes[2 * pair]) |
                                   (static_cast<std::uint16_t>(bytes[2 * pair + 1]) << 8);
        const __half2 values = __hmul2(decode_fp8_pair_half(table, bits), scale);
        out[pair] = *reinterpret_cast<const unsigned*>(&values);
    }
}

// Lookup-free FP8 E4M3 -> half pair decode, mirroring the external Marlin
// dequant<half2, kFE4M3fn, skip_flop=true> bit transform: the FP8 exp/mantissa plane shifts one
// bit into the half exp/mantissa position (FP16_EXPONENT - FP8_EXPONENT = 1) and the sign keeps
// its place. The 2^8 exponent bias is folded into the scale the caller multiplies in, so this is
// two mask/shift/or ALU ops per two codes with no shared table and no byte permute. The E4M3
// 0x7f/0xff NaN payloads map to +/-Inf instead of NaN; normal and subnormal codes are exact.
__device__ __forceinline__ __half2 decode_fp8_pair_alu(std::uint16_t packed) {
    unsigned q = packed;
    constexpr unsigned kMask = 0x7F007F00u;
    const unsigned first = (q & 0x80008000u) | ((q & kMask) >> 1);
    q <<= 8;
    const unsigned second = (q & 0x80008000u) | ((q & kMask) >> 1);
    const unsigned combined = (second & 0xFFFFu) | ((first & 0xFFFFu) << 16);
    return *reinterpret_cast<const __half2*>(&combined);
}

// Register-side ALU dequantization of one packed 16x32 unit, same shape as fp8_dequant_a_unit but
// with no lookup table. `scale` must already carry the 2^8 exponent bias (fp8_block_stage_scale_bias8).
__device__ __forceinline__ void fp8_dequant_a_unit_alu(uint4 raw, __half2 scale,
                                                       unsigned (&out)[8]) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const std::uint16_t bits = static_cast<std::uint16_t>(bytes[2 * pair]) |
                                   (static_cast<std::uint16_t>(bytes[2 * pair + 1]) << 8);
        const __half2 values = __hmul2(decode_fp8_pair_alu(bits), scale);
        out[pair] = *reinterpret_cast<const unsigned*>(&values);
    }
}

// Scale read that folds the FP8->FP16 exponent bias (2^8) into the BF16 multiplier, so the ALU
// decode above needs only the sign + one-bit-shift transform. BF16 x 256 is exact (exponent +8),
// and the half conversion widens the BF16 mantissa.
__device__ __forceinline__ __half2 fp8_block_stage_scale_bias8(const __nv_bfloat16* scales,
                                                               std::int64_t index, int lane) {
    unsigned scale_bits = 0;
    if (lane == 0) {
        scale_bits = __half_as_ushort(
            __float2half_rn(__bfloat162float(scales[index]) * 256.0f));
    }
    scale_bits = __shfl_sync(0xffffffffU, scale_bits, 0);
    return __half2half2(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));
}

// ALU-decode variant of stage_fp8_word: the same 16-byte store, but each pair decodes through
// decode_fp8_pair_alu (mask/shift/or, no shared table) with the 2^8 exponent bias folded into
// `scale` by fp8_block_stage_scale_bias8. Removes 8 LDS + 4 PRMT per staged word from the LSU at
// the cost of ~8-10 ALU ops per pair, for sm_75 where the LUT's shared loads contend with the
// staging stores and ldmatrix on the same LSU channel. The staged word bits are identical to the
// LUT route for normal and subnormal codes (0x7f/0xff NaN payloads map to +/-Inf instead).
__device__ __forceinline__ void stage_fp8_word_alu(__half* destination, uint2 packed,
                                                   __half2 scale) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
    unsigned words[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const std::uint16_t bits = static_cast<std::uint16_t>(bytes[2 * pair]) |
                                   (static_cast<std::uint16_t>(bytes[2 * pair + 1]) << 8);
        const __half2 values = __hmul2(decode_fp8_pair_alu(bits), scale);
        words[pair] = *reinterpret_cast<const unsigned*>(&values);
    }
    store_vec(destination, make_uint4(words[0], words[1], words[2], words[3]));
}

__device__ __forceinline__ void stage_bf16_word(__half* destination, int4 packed) {
    store_vec(destination, bf162_to_f162(static_cast<unsigned>(packed.x)));
    store_vec(destination + 2, bf162_to_f162(static_cast<unsigned>(packed.y)));
    store_vec(destination + 4, bf162_to_f162(static_cast<unsigned>(packed.z)));
    store_vec(destination + 6, bf162_to_f162(static_cast<unsigned>(packed.w)));
}

template <bool Broadcast>
__device__ __forceinline__ __half2 fp8_block_stage_scale(const __nv_bfloat16* scales,
                                                          std::int64_t index, int lane) {
    if constexpr (Broadcast) {
        unsigned scale_bits = 0;
        if (lane == 0) {
            scale_bits = __half_as_ushort(__float2half_rn(__bfloat162float(scales[index])));
        }
        scale_bits = __shfl_sync(0xffffffffU, scale_bits, 0);
        return __half2half2(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));
    }
    return __half2half2(__float2half_rn(__bfloat162float(scales[index])));
}

__device__ __forceinline__ void fp8_mma_f16_m16n8k8_f16acc(unsigned& c0, unsigned& c1,
                                                           unsigned a0, unsigned a1,
                                                           unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void fp8_mma_f16_f16acc(unsigned (&c)[2], unsigned a0, unsigned a1,
                                                   unsigned a2, unsigned a3, unsigned b0,
                                                   unsigned b1) {
    fp8_mma_f16_m16n8k8_f16acc(c[0], c[1], a0, a1, b0);
    fp8_mma_f16_m16n8k8_f16acc(c[0], c[1], a2, a3, b1);
}

__device__ __forceinline__ void fp8_fold_f16_partial(float (&destination)[4],
                                                     unsigned (&source)[2]) {
    const float2 low = __half22float2(*reinterpret_cast<const __half2*>(&source[0]));
    const float2 high = __half22float2(*reinterpret_cast<const __half2*>(&source[1]));
    destination[0] += low.x;
    destination[1] += low.y;
    destination[2] += high.x;
    destination[3] += high.y;
    source[0] = 0u;
    source[1] = 0u;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

template <class Output, bool Marlin, int TokensPerBlock>
__global__ __launch_bounds__(kThreads) void fp8_block_linear_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, Output output, int output_rows, int input_rows,
    int tokens) {
    constexpr int kTokensPerBlock = TokensPerBlock;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    const int token_begin = static_cast<int>(blockIdx.y) * kTokensPerBlock;
    const int k_tiles = input_rows / 128;
    __shared__ float fp8_code_table[kFp8CodeCount];
    build_fp8_code_table_float(fp8_code_table, static_cast<int>(threadIdx.x), kThreads);
    __syncthreads();
    // The block scale is uniform across the warp, so each lane can apply it to its own partial
    // and fold into a lane-local total. The warp reduction is then needed once, not once per
    // 128-wide K block.
    float totals[kTokensPerBlock] = {};

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int k_begin = k_tile * 128 + lane * 4;
        const std::uint32_t packed_codes =
            Marlin ? marlin_fp8_block_word(codes, row, k_begin, output_rows)
                   : *reinterpret_cast<const std::uint32_t*>(
                         codes + static_cast<std::int64_t>(row) * input_rows + k_begin);
        float2 weight01;
        float2 weight23;
#if NINFER_FP8_SCALAR_ABLATION == 1
        weight01 = decode_fp8_pair_ablated(packed_codes);
        weight23 = decode_fp8_pair_ablated(packed_codes >> 16);
#else
        weight01 = decode_fp8_pair(fp8_code_table, static_cast<std::uint16_t>(packed_codes));
        weight23 = decode_fp8_pair(fp8_code_table,
                                   static_cast<std::uint16_t>(packed_codes >> 16));
#endif
        const float scale = __bfloat162float(scales[fp8_block_scale_index(row, k_tile, k_tiles)]);
        const __nv_bfloat16* activation =
            x + static_cast<std::int64_t>(token_begin) * input_rows + k_begin;
#pragma unroll
        for (int token_slot = 0; token_slot < kTokensPerBlock; ++token_slot) {
            const int token = token_begin + token_slot;
            if (token < tokens) {
                const uint2 packed_activation =
                    load_ldg<uint2>(reinterpret_cast<const uint2*>(activation));
                const float2 activation01 = bf16x2_wide(packed_activation.x);
                const float2 activation23 = bf16x2_wide(packed_activation.y);
                float partial = 0.0F;
                partial = fmaf(weight01.x, activation01.x, partial);
                partial = fmaf(weight01.y, activation01.y, partial);
                partial = fmaf(weight23.x, activation23.x, partial);
                partial = fmaf(weight23.y, activation23.y, partial);
                totals[token_slot] = fmaf(scale, partial, totals[token_slot]);
            }
            activation += static_cast<std::int64_t>(input_rows);
        }
    }

#pragma unroll
    for (int token_slot = 0; token_slot < kTokensPerBlock; ++token_slot) {
        const int token = token_begin + token_slot;
        if (token < tokens) {
            const float tile_sum = warp_sum(totals[token_slot]);
            if (lane == 0) { output.store(row, token, tile_sum); }
        }
    }
}

template <class Output, int Stages, bool BroadcastScale, bool NoFold = false,
          bool AluDecode = false>
__global__ __launch_bounds__(kHmmaThreads) void fp8_block_linear_hmma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, Output output, int output_rows, int input_rows,
    int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    auto weights_at = [&](int stage, int offset) -> __half* {
        return shared + stage * (kHmmaBlockRows * kHmmaBlockK) + offset;
    };
    auto activations_at = [&](int stage, int offset) -> __half* {
        return shared + Stages * (kHmmaBlockRows * kHmmaBlockK) +
               stage * (kHmmaBlockCols * kHmmaBlockK) + offset;
    };

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp / (kHmmaBlockCols / kHmmaWarpCols);
    const int warp_col = warp % (kHmmaBlockCols / kHmmaWarpCols);
    const int row0 = static_cast<int>(blockIdx.x) * kHmmaBlockRows;
    const int col0 = static_cast<int>(blockIdx.y) * kHmmaBlockCols;
    const int a_matrix = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int scale_tiles = input_rows / 128;
    const int k_tiles = input_rows / kHmmaBlockK;

    const std::uint8_t* weight_sources[kHmmaItems];
    int weight_destinations[kHmmaItems];
    const __nv_bfloat16* activation_sources[kHmmaItems];
    int activation_destinations[kHmmaItems];
#pragma unroll
    for (int i = 0; i < kHmmaItems; ++i) {
        const int item = tid + i * kHmmaThreads;
        const int row = item / (kHmmaBlockK / 8);
        const int k8 = item - row * (kHmmaBlockK / 8);
        weight_sources[i] = codes + static_cast<std::int64_t>(row0 + row) * input_rows + k8 * 8;
        weight_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
        const int token = col0 + row;
        activation_sources[i] =
            x + static_cast<std::int64_t>(token) * input_rows + k8 * 8;
        activation_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
    }

    auto load_tile = [&](int tile, uint2 (&weight_prefetch)[kHmmaItems],
                         int4 (&activation_prefetch)[kHmmaItems]) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kHmmaItems; ++i) {
            weight_prefetch[i] = load_ldg<uint2>(weight_sources[i] + k0);
            const int token = col0 + (tid + i * kHmmaThreads) / (kHmmaBlockK / 8);
            activation_prefetch[i] = token < tokens
                                         ? load_ldg<int4>(activation_sources[i] + k0)
                                         : make_int4(0, 0, 0, 0);
        }
    };

    auto store_tile = [&](int stage, const uint2 (&weight_prefetch)[kHmmaItems],
                          const int4 (&activation_prefetch)[kHmmaItems], int tile) {
        const std::int64_t scale_index =
            static_cast<std::int64_t>(row0 / 128) * scale_tiles + tile / 2;
        const __half2 scale = [&]() {
            if constexpr (AluDecode) {
                return fp8_block_stage_scale_bias8(scales, scale_index, lane);
            } else {
                return fp8_block_stage_scale<BroadcastScale>(scales, scale_index, lane);
            }
        }();
#pragma unroll
        for (int i = 0; i < kHmmaItems; ++i) {
            if constexpr (AluDecode) {
                stage_fp8_word_alu(weights_at(stage, weight_destinations[i]),
                                   weight_prefetch[i], scale);
            } else {
                stage_fp8_word(fp8_code_table, weights_at(stage, weight_destinations[i]),
                               weight_prefetch[i], scale);
            }
            stage_bf16_word(activations_at(stage, activation_destinations[i]),
                            activation_prefetch[i]);
        }
    };

    float accum[kHmmaRows][kHmmaCols][4] = {};
    unsigned accum_h[kHmmaRows][kHmmaCols][2] = {};
    if constexpr (NoFold) {
        (void)accum;
    }
    if constexpr (!AluDecode) {
        build_fp8_code_table(fp8_code_table, tid);
        __syncthreads();
    }
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
        if (stage < k_tiles) {
            uint2 weight_prefetch[kHmmaItems];
            int4 activation_prefetch[kHmmaItems];
            load_tile(stage, weight_prefetch, activation_prefetch);
            store_tile(stage, weight_prefetch, activation_prefetch, stage);
        }
    }
    __syncthreads();

    for (int tile = 0; tile < k_tiles; ++tile) {
        const int stage = tile % Stages;
        uint2 weight_prefetch[kHmmaItems];
        int4 activation_prefetch[kHmmaItems];
        const int prefetch = tile + Stages;
        if (prefetch < k_tiles) { load_tile(prefetch, weight_prefetch, activation_prefetch); }

#pragma unroll
        for (int k_sub = 0; k_sub < kHmmaKSub; ++k_sub) {
            const int k_step = k_sub * 16;
            unsigned a_fragments[kHmmaRows][4];
            unsigned b_fragments[kHmmaCols][2];
#pragma unroll
            for (int mi = 0; mi < kHmmaRows; ++mi) {
                const int row = warp_row * kHmmaWarpRows + mi * 16 + a_row_offset;
                const int column = k_step + a_col_offset;
                ldmatrix_x4(a_fragments[mi][0], a_fragments[mi][1], a_fragments[mi][2],
                            a_fragments[mi][3],
                            smem_addr(weights_at(stage, row * kHmmaBlockK +
                                                  fp8_block_swizzle(row, column))));
            }
#pragma unroll
            for (int ni = 0; ni < kHmmaCols; ++ni) {
                const int row = warp_col * kHmmaWarpCols + ni * 8 + b_inner_row;
                const int column = k_step + b_k_offset;
                ldmatrix_x2(b_fragments[ni][0], b_fragments[ni][1],
                            smem_addr(activations_at(stage, row * kHmmaBlockK +
                                                      fp8_block_swizzle(row, column))));
            }
#pragma unroll
            for (int mi = 0; mi < kHmmaRows; ++mi) {
#pragma unroll
                for (int ni = 0; ni < kHmmaCols; ++ni) {
                    fp8_mma_f16_f16acc(accum_h[mi][ni], a_fragments[mi][0], a_fragments[mi][1],
                                       a_fragments[mi][2], a_fragments[mi][3],
                                       b_fragments[ni][0], b_fragments[ni][1]);
                }
            }
        }

        if constexpr (!NoFold) {
            for (int mi = 0; mi < kHmmaRows; ++mi) {
#pragma unroll
                for (int ni = 0; ni < kHmmaCols; ++ni) {
                    fp8_fold_f16_partial(accum[mi][ni], accum_h[mi][ni]);
                }
            }
        }

        __syncthreads();
        if (prefetch < k_tiles) {
            store_tile(stage, weight_prefetch, activation_prefetch, prefetch);
            // `stage == tile % Stages` is both the buffer just consumed and the buffer being
            // refilled, and `prefetch == tile + Stages` means it is read again on the very next
            // iteration when Stages == 1. The pre-store barrier only orders the reads that
            // precede it; without a barrier after the store a warp can reach the next
            // iteration's ldmatrix before another warp has finished writing its slice of the
            // activation tile. `prefetch < k_tiles` is block-uniform, so this barrier is too.
            __syncthreads();
        }
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kHmmaRows; ++mi) {
        const int row_a = row0 + warp_row * kHmmaWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kHmmaCols; ++ni) {
            const int col_a = col0 + warp_col * kHmmaWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            float values[4];
            if constexpr (NoFold) {
                const float2 low =
                    __half22float2(*reinterpret_cast<const __half2*>(&accum_h[mi][ni][0]));
                const float2 high =
                    __half22float2(*reinterpret_cast<const __half2*>(&accum_h[mi][ni][1]));
                values[0] = low.x;
                values[1] = low.y;
                values[2] = high.x;
                values[3] = high.y;
            } else {
                const float* source = accum[mi][ni];
                values[0] = source[0];
                values[1] = source[1];
                values[2] = source[2];
                values[3] = source[3];
            }
            if (row_a < output_rows) {
                if (col_a < tokens) { output.store(row_a, col_a, values[0]); }
                if (col_b < tokens) { output.store(row_a, col_b, values[1]); }
            }
            if (row_b < output_rows) {
                if (col_a < tokens) { output.store(row_b, col_a, values[2]); }
                if (col_b < tokens) { output.store(row_b, col_b, values[3]); }
            }
        }
    }
}

// The gate and up weight tiles are staged into one 256 row buffer and consumed by a single fused
// compute, so the activation tile is staged once, every activation fragment is loaded once instead
// of once per pass, and a K-tile costs two barriers instead of four. The kernel already sits at the
// 255 register ceiling (its two accumulator sets need 192), so nothing here may add registers.
//
// The SwiGLU leaf owns its block shape and its thread count; the linear leaf's shared constants are
// left alone because its own tiling is a separately validated local optimum. A CTA still owns 128
// intermediate rows, so its accumulator count, thread budget and shared footprint are unchanged,
// but it covers 256 tokens instead of 128. Every weight row block is therefore visited by half as
// many CTAs, which halves the packed weight traffic. That plane is what does not fit L2 -- a CTA
// walks 1.25 MB of it and 68 concurrent CTAs walk 89 MB against a 5.5 MB cache -- while the
// activation tile it shares with its row neighbours is reused 136 times per column block.
//
// Do not widen this tile to 16 warps. At this row/column ratio a 32x32 warp tile becomes a 2 x 8
// warp grid, which doubles the weight ldmatrix redundancy from 4x to 8x, and 512 threads put the
// register ceiling at 128 with 96 of those registers already committed to accumulators. Measured:
// 64100 us against this configuration's 32415 us.
constexpr int kSwiGluRows = 64;
constexpr int kSwiGluCols = 256;
constexpr int kSwiGluWarpRows = 32;
constexpr int kSwiGluWarpCols = 64;
constexpr int kSwiGluWarps = (kSwiGluRows / kSwiGluWarpRows) * (kSwiGluCols / kSwiGluWarpCols);
constexpr int kSwiGluThreads = kSwiGluWarps * 32;
constexpr int kSwiGluMRows = kSwiGluWarpRows / 16;
constexpr int kSwiGluNCols = kSwiGluWarpCols / 8;
constexpr int kSwiGluWeightItems = kSwiGluRows * (kHmmaBlockK / 8) / kSwiGluThreads;
constexpr int kSwiGluActItems = kSwiGluCols * (kHmmaBlockK / 8) / kSwiGluThreads;
static_assert(kSwiGluWeightItems * kSwiGluThreads == kSwiGluRows * (kHmmaBlockK / 8));
static_assert(kSwiGluActItems * kSwiGluThreads == kSwiGluCols * (kHmmaBlockK / 8));
// The swiglu dispatch admits a weight whose row count is a multiple of `2 * kBlockRows`; this makes
// that guard imply that `intermediate_rows` divides by `kSwiGluRows` with no remainder, so the row
// grid needs no tail case. Change either constant and the launcher has to grow one.
static_assert(kBlockRows % kSwiGluRows == 0);
constexpr int kSwiGluHmmaSharedBytes =
    2 * kSwiGluRows * kHmmaBlockK * static_cast<int>(sizeof(__half)) +
    kSwiGluCols * kHmmaBlockK * static_cast<int>(sizeof(__half));
static_assert(kSwiGluHmmaSharedBytes == 49152);
constexpr int kSwiGluRegdequantSharedBytes =
    kSwiGluCols * kHmmaBlockK * static_cast<int>(sizeof(__half));
static_assert(kSwiGluRegdequantSharedBytes == 32768);

template <bool BroadcastScale, bool NoFold = false, bool AluDecode = false>
__global__ __launch_bounds__(kSwiGluThreads) void fp8_block_swiglu_hmma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int intermediate_rows, int input_rows, int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    auto weights_at = [&](int offset) -> __half* { return shared + offset; };
    auto activations_at = [&](int offset) -> __half* {
        return shared + 2 * (kSwiGluRows * kHmmaBlockK) + offset;
    };

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp / (kSwiGluCols / kSwiGluWarpCols);
    const int warp_col = warp % (kSwiGluCols / kSwiGluWarpCols);
    const int row0 = static_cast<int>(blockIdx.x) * kSwiGluRows;
    const int col0 = static_cast<int>(blockIdx.y) * kSwiGluCols;
    const int a_matrix = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int scale_tiles = input_rows / 128;
    const int k_tiles = input_rows / kHmmaBlockK;

    const std::uint8_t* gate_sources[kSwiGluWeightItems];
    const std::uint8_t* up_sources[kSwiGluWeightItems];
    int weight_destinations[kSwiGluWeightItems];
    const __nv_bfloat16* activation_sources[kSwiGluActItems];
    int activation_destinations[kSwiGluActItems];
 #pragma unroll
    for (int i = 0; i < kSwiGluWeightItems; ++i) {
        const int item = tid + i * kSwiGluThreads;
        const int row = item / (kHmmaBlockK / 8);
        const int k8 = item - row * (kHmmaBlockK / 8);
        gate_sources[i] = codes + static_cast<std::int64_t>(row0 + row) * input_rows + k8 * 8;
        up_sources[i] = codes +
                        static_cast<std::int64_t>(row0 + intermediate_rows + row) * input_rows +
                        k8 * 8;
        weight_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
    }
 #pragma unroll
    for (int i = 0; i < kSwiGluActItems; ++i) {
        const int item = tid + i * kSwiGluThreads;
        const int row = item / (kHmmaBlockK / 8);
        const int k8 = item - row * (kHmmaBlockK / 8);
        const int token = col0 + row;
        activation_sources[i] = x + static_cast<std::int64_t>(token) * input_rows + k8 * 8;
        activation_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
    }

    auto stage_activation = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kSwiGluActItems; ++i) {
            const int token = col0 + (tid + i * kSwiGluThreads) / (kHmmaBlockK / 8);
            const int4 packed = token < tokens
                                    ? load_ldg<int4>(activation_sources[i] + k0)
                                    : make_int4(0, 0, 0, 0);
            stage_bf16_word(activations_at(activation_destinations[i]), packed);
        }
    };

    auto stage_weights = [&](const std::uint8_t* const (&sources)[kSwiGluWeightItems], int tile,
                             int row_offset) {
        const std::int64_t scale_index =
            static_cast<std::int64_t>((row0 + row_offset) / 128) * scale_tiles + tile / 2;
        const __half2 scale = [&]() {
            if constexpr (AluDecode) {
                return fp8_block_stage_scale_bias8(scales, scale_index, lane);
            } else {
                return fp8_block_stage_scale<BroadcastScale>(scales, scale_index, lane);
            }
        }();
        const int k0 = tile * kHmmaBlockK;
        // The up half of the weight tile lives one full tile above the gate half; `row_offset` is
        // either zero or `intermediate_rows`, and only the latter lands in the second half.
        const int row_base = (row_offset == 0) ? 0 : kSwiGluRows * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kSwiGluWeightItems; ++i) {
            const uint2 packed = load_ldg<uint2>(sources[i] + k0);
            if constexpr (AluDecode) {
                stage_fp8_word_alu(weights_at(weight_destinations[i] + row_base), packed, scale);
            } else {
                stage_fp8_word(fp8_code_table, weights_at(weight_destinations[i] + row_base),
                               packed, scale);
            }
        }
    };

    auto compute = [&](unsigned (&gate_partial)[kSwiGluMRows][kSwiGluNCols][2],
                       unsigned (&up_partial)[kSwiGluMRows][kSwiGluNCols][2]) {
#pragma unroll
        for (int k_sub = 0; k_sub < kHmmaKSub; ++k_sub) {
            const int k_step = k_sub * 16;
            unsigned b_fragments[kSwiGluNCols][2];
#pragma unroll
            for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                const int row = warp_col * kSwiGluWarpCols + ni * 8 + b_inner_row;
                const int column = k_step + b_k_offset;
                ldmatrix_x2(b_fragments[ni][0], b_fragments[ni][1],
                            smem_addr(activations_at(row * kHmmaBlockK +
                                                      fp8_block_swizzle(row, column))));
            }
            // The activation fragments are shared by the two halves, so the second half of the
            // weight tile costs one ldmatrix_x4 per m16 row instead of a whole second pass.
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                unsigned a_fragments[kSwiGluMRows][4];
                unsigned (&partial)[kSwiGluMRows][kSwiGluNCols][2] =
                    (half == 0) ? gate_partial : up_partial;
#pragma unroll
                for (int mi = 0; mi < kSwiGluMRows; ++mi) {
                    const int row = half * kSwiGluRows + warp_row * kSwiGluWarpRows +
                                    mi * 16 + a_row_offset;
                    const int column = k_step + a_col_offset;
                    ldmatrix_x4(a_fragments[mi][0], a_fragments[mi][1], a_fragments[mi][2],
                                a_fragments[mi][3],
                                smem_addr(weights_at(row * kHmmaBlockK +
                                                     fp8_block_swizzle(row, column))));
                }
#pragma unroll
                for (int mi = 0; mi < kSwiGluMRows; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                        fp8_mma_f16_f16acc(partial[mi][ni], a_fragments[mi][0],
                                           a_fragments[mi][1], a_fragments[mi][2],
                                           a_fragments[mi][3], b_fragments[ni][0],
                                           b_fragments[ni][1]);
                    }
                }
            }
        }
    };

    float gate_accum[kSwiGluMRows][kSwiGluNCols][4] = {};
    float up_accum[kSwiGluMRows][kSwiGluNCols][4] = {};
    unsigned gate_partial[kSwiGluMRows][kSwiGluNCols][2] = {};
    unsigned up_partial[kSwiGluMRows][kSwiGluNCols][2] = {};
    if constexpr (NoFold) {
        (void)gate_accum;
        (void)up_accum;
    }

    if constexpr (!AluDecode) {
        build_fp8_code_table(fp8_code_table, tid);
        __syncthreads();
    }

    for (int tile = 0; tile < k_tiles; ++tile) {
        stage_activation(tile);
        stage_weights(gate_sources, tile, 0);
        stage_weights(up_sources, tile, intermediate_rows);
        __syncthreads();
        compute(gate_partial, up_partial);
        if constexpr (!NoFold) {
#pragma unroll
            for (int mi = 0; mi < kSwiGluMRows; ++mi) {
#pragma unroll
                for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                    fp8_fold_f16_partial(gate_accum[mi][ni], gate_partial[mi][ni]);
                    fp8_fold_f16_partial(up_accum[mi][ni], up_partial[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kSwiGluMRows; ++mi) {
        const int row_a = row0 + warp_row * kSwiGluWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kSwiGluNCols; ++ni) {
            const int col_a = col0 + warp_col * kSwiGluWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            float gate[4];
            float up[4];
            if constexpr (NoFold) {
                const float2 gate_low =
                    __half22float2(*reinterpret_cast<const __half2*>(&gate_partial[mi][ni][0]));
                const float2 gate_high =
                    __half22float2(*reinterpret_cast<const __half2*>(&gate_partial[mi][ni][1]));
                const float2 up_low =
                    __half22float2(*reinterpret_cast<const __half2*>(&up_partial[mi][ni][0]));
                const float2 up_high =
                    __half22float2(*reinterpret_cast<const __half2*>(&up_partial[mi][ni][1]));
                gate[0] = gate_low.x;
                gate[1] = gate_low.y;
                gate[2] = gate_high.x;
                gate[3] = gate_high.y;
                up[0] = up_low.x;
                up[1] = up_low.y;
                up[2] = up_high.x;
                up[3] = up_high.y;
            } else {
                const float* gate_source = gate_accum[mi][ni];
                const float* up_source = up_accum[mi][ni];
                gate[0] = gate_source[0];
                gate[1] = gate_source[1];
                gate[2] = gate_source[2];
                gate[3] = gate_source[3];
                up[0] = up_source[0];
                up[1] = up_source[1];
                up[2] = up_source[2];
                up[3] = up_source[3];
            }
            if (row_a < intermediate_rows) {
                if (col_a < tokens) {
                    out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_a] =
                        __float2bfloat16_rn(silu(gate[0]) * up[0]);
                }
                if (col_b < tokens) {
                    out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_a] =
                        __float2bfloat16_rn(silu(gate[1]) * up[1]);
                }
            }
            if (row_b < intermediate_rows) {
                if (col_a < tokens) {
                    out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_b] =
                        __float2bfloat16_rn(silu(gate[2]) * up[2]);
                }
                if (col_b < tokens) {
                    out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_b] =
                        __float2bfloat16_rn(silu(gate[3]) * up[3]);
                }
            }
        }
    }
}

// Register-dequantized SwiGLU HMMA prototype. The weight A fragments are decoded straight from
// global memory instead of staged through shared + ldmatrix, while the activation B fragments keep
// the shared staging path because they are small and reused across the two weight halves. The
// 32 KiB weight staging plane (STS.128 + decode + LDS) shares the LSU channel with ldmatrix on
// sm_75, so removing it tests whether the tensor pipe can be fed without that contention. Routed
// by NINFER_FP8_BLOCK_REGDEQUANT; not a committed replacement for the staged kernel.
template <bool BroadcastScale>
__global__ __launch_bounds__(kSwiGluThreads) void fp8_block_swiglu_hmma_regdequant_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int intermediate_rows, int input_rows, int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    auto activations_at = [&](int offset) -> __half* { return shared + offset; };

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp / (kSwiGluCols / kSwiGluWarpCols);
    const int warp_col = warp % (kSwiGluCols / kSwiGluWarpCols);
    const int row0 = static_cast<int>(blockIdx.x) * kSwiGluRows;
    const int col0 = static_cast<int>(blockIdx.y) * kSwiGluCols;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int scale_tiles = input_rows / 128;
    const int k_tiles = input_rows / kHmmaBlockK;
    const int row_a = lane >> 2;
    const int col_k = 2 * (lane & 3);

    const __nv_bfloat16* activation_sources[kSwiGluActItems];
    int activation_destinations[kSwiGluActItems];
 #pragma unroll
    for (int i = 0; i < kSwiGluActItems; ++i) {
        const int item = tid + i * kSwiGluThreads;
        const int row = item / (kHmmaBlockK / 8);
        const int k8 = item - row * (kHmmaBlockK / 8);
        const int token = col0 + row;
        activation_sources[i] = x + static_cast<std::int64_t>(token) * input_rows + k8 * 8;
        activation_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
    }

    auto stage_activation = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kSwiGluActItems; ++i) {
            const int token = col0 + (tid + i * kSwiGluThreads) / (kHmmaBlockK / 8);
            const int4 packed = token < tokens
                                    ? load_ldg<int4>(activation_sources[i] + k0)
                                    : make_int4(0, 0, 0, 0);
            stage_bf16_word(activations_at(activation_destinations[i]), packed);
        }
    };

    float gate_accum[kSwiGluMRows][kSwiGluNCols][4] = {};
    float up_accum[kSwiGluMRows][kSwiGluNCols][4] = {};
    unsigned gate_partial[kSwiGluMRows][kSwiGluNCols][2] = {};
    unsigned up_partial[kSwiGluMRows][kSwiGluNCols][2] = {};

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    for (int tile = 0; tile < k_tiles; ++tile) {
        stage_activation(tile);
        __syncthreads();
        const int k0 = tile * kHmmaBlockK;
        const __half2 gate_scale = fp8_block_stage_scale<BroadcastScale>(
            scales, static_cast<std::int64_t>(row0 / 128) * scale_tiles + tile / 2, lane);
        const __half2 up_scale = fp8_block_stage_scale<BroadcastScale>(
            scales, static_cast<std::int64_t>((row0 + intermediate_rows) / 128) * scale_tiles +
                        tile / 2,
            lane);
#pragma unroll
        for (int k_sub = 0; k_sub < kHmmaKSub; ++k_sub) {
            const int k_step = k_sub * 16;
            unsigned b_fragments[kSwiGluNCols][2];
#pragma unroll
            for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                const int row = warp_col * kSwiGluWarpCols + ni * 8 + b_inner_row;
                const int column = k_step + b_k_offset;
                ldmatrix_x2(b_fragments[ni][0], b_fragments[ni][1],
                            smem_addr(activations_at(row * kHmmaBlockK +
                                                      fp8_block_swizzle(row, column))));
            }
            // A fragments decoded in registers: the m16k8 lane owns four m16k8 registers. a0/a2
            // sit on row (lane>>2), a1/a3 on row 8+(lane>>2); a2/a3 take the +8 k pair. The k
            // within each pair is 2*(lane&3), matching the mma.m16n8k8 A-fragment layout.
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                unsigned (&partial)[kSwiGluMRows][kSwiGluNCols][2] =
                    (half == 0) ? gate_partial : up_partial;
                const __half2 scale = (half == 0) ? gate_scale : up_scale;
                const int half_row = half * intermediate_rows;
#pragma unroll
                for (int mi = 0; mi < kSwiGluMRows; ++mi) {
                    const int base_row =
                        row0 + half_row + warp_row * kSwiGluWarpRows + mi * 16;
                    const int kk = k0 + k_step;
                    unsigned a_fragments[4];
                    a_fragments[0] =
                        fp8_dequant_a_word(fp8_code_table, codes, base_row + row_a,
                                           kk + col_k, input_rows, scale);
                    a_fragments[1] =
                        fp8_dequant_a_word(fp8_code_table, codes, base_row + 8 + row_a,
                                           kk + col_k, input_rows, scale);
                    a_fragments[2] =
                        fp8_dequant_a_word(fp8_code_table, codes, base_row + row_a,
                                           kk + 8 + col_k, input_rows, scale);
                    a_fragments[3] =
                        fp8_dequant_a_word(fp8_code_table, codes, base_row + 8 + row_a,
                                           kk + 8 + col_k, input_rows, scale);
#pragma unroll
                    for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                        fp8_mma_f16_f16acc(partial[mi][ni], a_fragments[0], a_fragments[1],
                                           a_fragments[2], a_fragments[3], b_fragments[ni][0],
                                           b_fragments[ni][1]);
                    }
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < kSwiGluMRows; ++mi) {
#pragma unroll
            for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                fp8_fold_f16_partial(gate_accum[mi][ni], gate_partial[mi][ni]);
                fp8_fold_f16_partial(up_accum[mi][ni], up_partial[mi][ni]);
            }
        }
        __syncthreads();
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kSwiGluMRows; ++mi) {
        const int row_a = row0 + warp_row * kSwiGluWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kSwiGluNCols; ++ni) {
            const int col_a = col0 + warp_col * kSwiGluWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            const float* gate = gate_accum[mi][ni];
            const float* up = up_accum[mi][ni];
            if (row_a < intermediate_rows) {
                if (col_a < tokens) {
                    out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_a] =
                        __float2bfloat16_rn(silu(gate[0]) * up[0]);
                }
                if (col_b < tokens) {
                    out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_a] =
                        __float2bfloat16_rn(silu(gate[1]) * up[1]);
                }
            }
            if (row_b < intermediate_rows) {
                if (col_a < tokens) {
                    out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_b] =
                        __float2bfloat16_rn(silu(gate[2]) * up[2]);
                }
                if (col_b < tokens) {
                    out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_b] =
                        __float2bfloat16_rn(silu(gate[3]) * up[3]);
                }
            }
        }
    }
}

// Packed-layout (16x32 unit) register-dequantized SwiGLU HMMA. The weight codes are stored in a
// layout where each lane's 16 codes for two consecutive k-steps are contiguous, so one LDG.128 per
// lane per half per m16 row feeds two k-steps of mma with no shared staging and no ldmatrix on the
// weight side. The activation B fragments keep the shared staging path. This is the milestone-1
// replacement target once the converter writes the packed layout; it is routed by
// NINFER_FP8_BLOCK_PACKED while under development.
template <bool BroadcastScale>
__global__ __launch_bounds__(kSwiGluThreads) void fp8_block_swiglu_hmma_packed_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int intermediate_rows, int input_rows, int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    auto activations_at = [&](int offset) -> __half* { return shared + offset; };

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp / (kSwiGluCols / kSwiGluWarpCols);
    const int warp_col = warp % (kSwiGluCols / kSwiGluWarpCols);
    const int row0 = static_cast<int>(blockIdx.x) * kSwiGluRows;
    const int col0 = static_cast<int>(blockIdx.y) * kSwiGluCols;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int scale_tiles = input_rows / 128;
    const int k_tiles = input_rows / kHmmaBlockK;
    const int units_per_row = input_rows / 32;

    const __nv_bfloat16* activation_sources[kSwiGluActItems];
    int activation_destinations[kSwiGluActItems];
 #pragma unroll
    for (int i = 0; i < kSwiGluActItems; ++i) {
        const int item = tid + i * kSwiGluThreads;
        const int row = item / (kHmmaBlockK / 8);
        const int k8 = item - row * (kHmmaBlockK / 8);
        const int token = col0 + row;
        activation_sources[i] = x + static_cast<std::int64_t>(token) * input_rows + k8 * 8;
        activation_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
    }

    auto stage_activation = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kSwiGluActItems; ++i) {
            const int token = col0 + (tid + i * kSwiGluThreads) / (kHmmaBlockK / 8);
            const int4 packed = token < tokens
                                    ? load_ldg<int4>(activation_sources[i] + k0)
                                    : make_int4(0, 0, 0, 0);
            stage_bf16_word(activations_at(activation_destinations[i]), packed);
        }
    };

    float gate_accum[kSwiGluMRows][kSwiGluNCols][4] = {};
    float up_accum[kSwiGluMRows][kSwiGluNCols][4] = {};
    unsigned gate_partial[kSwiGluMRows][kSwiGluNCols][2] = {};
    unsigned up_partial[kSwiGluMRows][kSwiGluNCols][2] = {};

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    for (int tile = 0; tile < k_tiles; ++tile) {
        stage_activation(tile);
        __syncthreads();
        const int k0 = tile * kHmmaBlockK;
        const __half2 gate_scale = fp8_block_stage_scale<BroadcastScale>(
            scales, static_cast<std::int64_t>(row0 / 128) * scale_tiles + tile / 2, lane);
        const __half2 up_scale = fp8_block_stage_scale<BroadcastScale>(
            scales, static_cast<std::int64_t>((row0 + intermediate_rows) / 128) * scale_tiles +
                        tile / 2,
            lane);
#pragma unroll
        for (int k_pair = 0; k_pair < 2; ++k_pair) {
            const int unit_k = k0 + k_pair * 32;
            unsigned b_fragments[2][kSwiGluNCols][2];
#pragma unroll
            for (int step = 0; step < 2; ++step) {
                const int k_step = k_pair * 32 + step * 16;
#pragma unroll
                for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                    const int row = warp_col * kSwiGluWarpCols + ni * 8 + b_inner_row;
                    const int column = k_step + b_k_offset;
                    ldmatrix_x2(b_fragments[step][ni][0], b_fragments[step][ni][1],
                                smem_addr(activations_at(row * kHmmaBlockK +
                                                          fp8_block_swizzle(row, column))));
                }
            }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                unsigned (&partial)[kSwiGluMRows][kSwiGluNCols][2] =
                    (half == 0) ? gate_partial : up_partial;
                const __half2 scale = (half == 0) ? gate_scale : up_scale;
                const int half_row = half * intermediate_rows;
#pragma unroll
                for (int mi = 0; mi < kSwiGluMRows; ++mi) {
                    const int unit_row =
                        row0 + half_row + warp_row * kSwiGluWarpRows + mi * 16;
                    const int unit_index =
                        (unit_row / 16) * units_per_row + (unit_k / 32);
                    const uint4 raw = load_ldg<uint4>(
                        reinterpret_cast<const uint4*>(codes) + unit_index * 32 + lane);
                    unsigned a_unit[8];
                    fp8_dequant_a_unit(fp8_code_table, raw, scale, a_unit);
#pragma unroll
                    for (int step = 0; step < 2; ++step) {
#pragma unroll
                        for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                            fp8_mma_f16_f16acc(partial[mi][ni], a_unit[step * 4],
                                               a_unit[step * 4 + 1], a_unit[step * 4 + 2],
                                               a_unit[step * 4 + 3], b_fragments[step][ni][0],
                                               b_fragments[step][ni][1]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < kSwiGluMRows; ++mi) {
#pragma unroll
            for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                fp8_fold_f16_partial(gate_accum[mi][ni], gate_partial[mi][ni]);
                fp8_fold_f16_partial(up_accum[mi][ni], up_partial[mi][ni]);
            }
        }
        __syncthreads();
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kSwiGluMRows; ++mi) {
        const int row_a = row0 + warp_row * kSwiGluWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kSwiGluNCols; ++ni) {
            const int col_a = col0 + warp_col * kSwiGluWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            const float* gate = gate_accum[mi][ni];
            const float* up = up_accum[mi][ni];
            if (row_a < intermediate_rows) {
                if (col_a < tokens) {
                    out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_a] =
                        __float2bfloat16_rn(silu(gate[0]) * up[0]);
                }
                if (col_b < tokens) {
                    out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_a] =
                        __float2bfloat16_rn(silu(gate[1]) * up[1]);
                }
            }
            if (row_b < intermediate_rows) {
                if (col_a < tokens) {
                    out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_b] =
                        __float2bfloat16_rn(silu(gate[2]) * up[2]);
                }
                if (col_b < tokens) {
                    out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_b] =
                        __float2bfloat16_rn(silu(gate[3]) * up[3]);
                }
            }
        }
    }
}

// Broadcast-layout SwiGLU HMMA (milestone-1 scheme C): the raw weight codes are staged into
// shared memory once per block in the packed 16x32 layout (block-level, no warp-col redundancy in
// LDG), then every warp re-reads its own rows from shared with LDS.128 and decodes them in
// registers straight into mma A fragments. This keeps the 1x DRAM weight traffic of the staged
// kernel while dropping the STS.128-of-decoded + ldmatrix weight path, replacing it with a smaller
// raw-code STS and per-warp register decode. The activation B fragments keep the shared staging
// path. Routed by NINFER_FP8_BLOCK_BROADCAST while under development.
constexpr int kSwiGluCodeBytes = kSwiGluRows * kHmmaBlockK * static_cast<int>(sizeof(std::uint8_t));
constexpr int kSwiGluBroadcastSharedBytes =
    2 * kSwiGluCodeBytes + kSwiGluCols * kHmmaBlockK * static_cast<int>(sizeof(__half));
static_assert(kSwiGluBroadcastSharedBytes == 40960);

template <bool BroadcastScale>
__global__ __launch_bounds__(kSwiGluThreads) void fp8_block_swiglu_hmma_broadcast_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int intermediate_rows, int input_rows, int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    auto codes_at = [&](int offset) -> std::uint8_t* {
        return reinterpret_cast<std::uint8_t*>(shared) + offset;
    };
    auto activations_at = [&](int offset) -> __half* {
        return shared + (2 * kSwiGluCodeBytes) / static_cast<int>(sizeof(__half)) + offset;
    };

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp / (kSwiGluCols / kSwiGluWarpCols);
    const int warp_col = warp % (kSwiGluCols / kSwiGluWarpCols);
    const int row0 = static_cast<int>(blockIdx.x) * kSwiGluRows;
    const int col0 = static_cast<int>(blockIdx.y) * kSwiGluCols;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int scale_tiles = input_rows / 128;
    const int k_tiles = input_rows / kHmmaBlockK;

    const __nv_bfloat16* activation_sources[kSwiGluActItems];
    int activation_destinations[kSwiGluActItems];
 #pragma unroll
    for (int i = 0; i < kSwiGluActItems; ++i) {
        const int item = tid + i * kSwiGluThreads;
        const int row = item / (kHmmaBlockK / 8);
        const int k8 = item - row * (kHmmaBlockK / 8);
        const int token = col0 + row;
        activation_sources[i] = x + static_cast<std::int64_t>(token) * input_rows + k8 * 8;
        activation_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
    }

    auto stage_activation = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kSwiGluActItems; ++i) {
            const int token = col0 + (tid + i * kSwiGluThreads) / (kHmmaBlockK / 8);
            const int4 packed = token < tokens
                                    ? load_ldg<int4>(activation_sources[i] + k0)
                                    : make_int4(0, 0, 0, 0);
            stage_bf16_word(activations_at(activation_destinations[i]), packed);
        }
    };

    // Reorder the raw row-major codes of this K tile into the packed 16x32 shared layout. 512
    // lane-units (16 units x 32 lanes) split across the block; each thread does two units, reading
    // eight 2-code pairs from the row-major plane and writing one 16-code vector to shared.
    auto stage_codes = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int lu = tid + i * kSwiGluThreads;
            const int half = lu >> 8;
            const int unit_lane = lu & 255;
            const int unit = unit_lane >> 5;
            const int st_lane = unit_lane & 31;
            const int unit_row_tile = unit >> 1;
            const int unit_k_tile = unit & 1;
            const int unit_row = row0 + half * intermediate_rows + unit_row_tile * 16;
            const int unit_k = k0 + unit_k_tile * 32;
            const int r = st_lane >> 2;
            const int c = 2 * (st_lane & 3);
            const std::uint8_t* src =
                codes + static_cast<std::int64_t>(unit_row) * input_rows + unit_k;
            const std::uint16_t p0 =
                *reinterpret_cast<const std::uint16_t*>(src + static_cast<std::int64_t>(r) * input_rows + c);
            const std::uint16_t p1 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(8 + r) * input_rows + c);
            const std::uint16_t p2 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(r) * input_rows + 8 + c);
            const std::uint16_t p3 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(8 + r) * input_rows + 8 + c);
            const std::uint16_t p4 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(r) * input_rows + 16 + c);
            const std::uint16_t p5 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(8 + r) * input_rows + 16 + c);
            const std::uint16_t p6 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(r) * input_rows + 16 + 8 + c);
            const std::uint16_t p7 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(8 + r) * input_rows + 16 + 8 + c);
            const uint4 packed = make_uint4(
                static_cast<unsigned>(p0) | (static_cast<unsigned>(p1) << 16),
                static_cast<unsigned>(p2) | (static_cast<unsigned>(p3) << 16),
                static_cast<unsigned>(p4) | (static_cast<unsigned>(p5) << 16),
                static_cast<unsigned>(p6) | (static_cast<unsigned>(p7) << 16));
            store_vec(reinterpret_cast<uint4*>(
                          codes_at(half * kSwiGluCodeBytes + unit * 512 + st_lane * 16)),
                      packed);
        }
    };

    float gate_accum[kSwiGluMRows][kSwiGluNCols][4] = {};
    float up_accum[kSwiGluMRows][kSwiGluNCols][4] = {};
    unsigned gate_partial[kSwiGluMRows][kSwiGluNCols][2] = {};
    unsigned up_partial[kSwiGluMRows][kSwiGluNCols][2] = {};

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    for (int tile = 0; tile < k_tiles; ++tile) {
        stage_activation(tile);
        stage_codes(tile);
        __syncthreads();
        const __half2 gate_scale = fp8_block_stage_scale<BroadcastScale>(
            scales, static_cast<std::int64_t>(row0 / 128) * scale_tiles + tile / 2, lane);
        const __half2 up_scale = fp8_block_stage_scale<BroadcastScale>(
            scales, static_cast<std::int64_t>((row0 + intermediate_rows) / 128) * scale_tiles +
                        tile / 2,
            lane);
#pragma unroll
        for (int k_pair = 0; k_pair < 2; ++k_pair) {
            unsigned b_fragments[2][kSwiGluNCols][2];
#pragma unroll
            for (int step = 0; step < 2; ++step) {
                const int k_step = k_pair * 32 + step * 16;
#pragma unroll
                for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                    const int row = warp_col * kSwiGluWarpCols + ni * 8 + b_inner_row;
                    const int column = k_step + b_k_offset;
                    ldmatrix_x2(b_fragments[step][ni][0], b_fragments[step][ni][1],
                                smem_addr(activations_at(row * kHmmaBlockK +
                                                          fp8_block_swizzle(row, column))));
                }
            }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                unsigned (&partial)[kSwiGluMRows][kSwiGluNCols][2] =
                    (half == 0) ? gate_partial : up_partial;
                const __half2 scale = (half == 0) ? gate_scale : up_scale;
#pragma unroll
                for (int mi = 0; mi < kSwiGluMRows; ++mi) {
                    const int unit = (warp_row * 2 + mi) * 2 + k_pair;
                    const uint4 raw = *reinterpret_cast<const uint4*>(
                        codes_at(half * kSwiGluCodeBytes + unit * 512 + lane * 16));
                    unsigned a_unit[8];
                    fp8_dequant_a_unit(fp8_code_table, raw, scale, a_unit);
#pragma unroll
                    for (int step = 0; step < 2; ++step) {
#pragma unroll
                        for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                            fp8_mma_f16_f16acc(partial[mi][ni], a_unit[step * 4],
                                               a_unit[step * 4 + 1], a_unit[step * 4 + 2],
                                               a_unit[step * 4 + 3], b_fragments[step][ni][0],
                                               b_fragments[step][ni][1]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < kSwiGluMRows; ++mi) {
#pragma unroll
            for (int ni = 0; ni < kSwiGluNCols; ++ni) {
                fp8_fold_f16_partial(gate_accum[mi][ni], gate_partial[mi][ni]);
                fp8_fold_f16_partial(up_accum[mi][ni], up_partial[mi][ni]);
            }
        }
        __syncthreads();
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kSwiGluMRows; ++mi) {
        const int row_a = row0 + warp_row * kSwiGluWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kSwiGluNCols; ++ni) {
            const int col_a = col0 + warp_col * kSwiGluWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            const float* gate = gate_accum[mi][ni];
            const float* up = up_accum[mi][ni];
            if (row_a < intermediate_rows) {
                if (col_a < tokens) {
                    out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_a] =
                        __float2bfloat16_rn(silu(gate[0]) * up[0]);
                }
                if (col_b < tokens) {
                    out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_a] =
                        __float2bfloat16_rn(silu(gate[1]) * up[1]);
                }
            }
            if (row_b < intermediate_rows) {
                if (col_a < tokens) {
                    out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_b] =
                        __float2bfloat16_rn(silu(gate[2]) * up[2]);
                }
                if (col_b < tokens) {
                    out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_b] =
                        __float2bfloat16_rn(silu(gate[3]) * up[3]);
                }
            }
        }
    }
}

// Small-tile (64x64 token) SwiGLU HMMA. Four warps own four disjoint 16-row bands, so there is no
// warp-col weight redundancy: each warp reads its own 16 rows exactly once per K tile. The raw
// codes are staged into shared in the packed 16x32 layout at block level, then each warp decodes
// its own rows in registers straight into mma A fragments. This trades the large-tile weight
// sharing (fewer DRAM reads) for zero LDSM on the weight side and no per-warp redundancy; the
// smaller tile also cuts the activation footprint to 8 KiB. Routed by NINFER_FP8_BLOCK_SMALLTILE.
constexpr int kSmallTileCols = 64;
constexpr int kSmallTileWarps = 4;
constexpr int kSmallTileThreads = kSmallTileWarps * 32;
constexpr int kSmallTileWarpRows = 16;
constexpr int kSmallTileNCols = kSmallTileCols / 8;
constexpr int kSmallTileActItems = kSmallTileCols * (kHmmaBlockK / 8) / kSmallTileThreads;
constexpr int kSmallTileCodeBytes = kSwiGluRows * kHmmaBlockK;
constexpr int kSmallTileSharedBytes =
    2 * kSmallTileCodeBytes + kSmallTileCols * kHmmaBlockK * static_cast<int>(sizeof(__half));
static_assert(kSmallTileThreads * kSmallTileActItems == kSmallTileCols * (kHmmaBlockK / 8));
static_assert(kSmallTileSharedBytes == 16384);

template <bool BroadcastScale>
__global__ __launch_bounds__(kSmallTileThreads) void fp8_block_swiglu_hmma_smalltile_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int intermediate_rows, int input_rows, int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    auto codes_at = [&](int offset) -> std::uint8_t* {
        return reinterpret_cast<std::uint8_t*>(shared) + offset;
    };
    auto activations_at = [&](int offset) -> __half* {
        return shared + (2 * kSmallTileCodeBytes) / static_cast<int>(sizeof(__half)) + offset;
    };

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = static_cast<int>(blockIdx.x) * kSwiGluRows;
    const int col0 = static_cast<int>(blockIdx.y) * kSmallTileCols;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int scale_tiles = input_rows / 128;
    const int k_tiles = input_rows / kHmmaBlockK;

    const __nv_bfloat16* activation_sources[kSmallTileActItems];
    int activation_destinations[kSmallTileActItems];
 #pragma unroll
    for (int i = 0; i < kSmallTileActItems; ++i) {
        const int item = tid + i * kSmallTileThreads;
        const int token_row = item / (kHmmaBlockK / 8);
        const int k8 = item - token_row * (kHmmaBlockK / 8);
        const int token = col0 + token_row;
        activation_sources[i] = x + static_cast<std::int64_t>(token) * input_rows + k8 * 8;
        activation_destinations[i] = token_row * kHmmaBlockK + fp8_block_swizzle(token_row, k8 * 8);
    }

    auto stage_activation = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kSmallTileActItems; ++i) {
            const int token = col0 + (tid + i * kSmallTileThreads) / (kHmmaBlockK / 8);
            const int4 packed = token < tokens
                                    ? load_ldg<int4>(activation_sources[i] + k0)
                                    : make_int4(0, 0, 0, 0);
            stage_bf16_word(activations_at(activation_destinations[i]), packed);
        }
    };

    auto stage_codes = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lu = tid + i * kSmallTileThreads;
            const int half = lu >> 8;
            const int unit_lane = lu & 255;
            const int unit = unit_lane >> 5;
            const int st_lane = unit_lane & 31;
            const int unit_row_tile = unit >> 1;
            const int unit_k_tile = unit & 1;
            const int unit_row = row0 + half * intermediate_rows + unit_row_tile * 16;
            const int unit_k = k0 + unit_k_tile * 32;
            const int r = st_lane >> 2;
            const int c = 2 * (st_lane & 3);
            const std::uint8_t* src =
                codes + static_cast<std::int64_t>(unit_row) * input_rows + unit_k;
            const std::uint16_t p0 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(r) * input_rows + c);
            const std::uint16_t p1 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(8 + r) * input_rows + c);
            const std::uint16_t p2 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(r) * input_rows + 8 + c);
            const std::uint16_t p3 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(8 + r) * input_rows + 8 + c);
            const std::uint16_t p4 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(r) * input_rows + 16 + c);
            const std::uint16_t p5 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(8 + r) * input_rows + 16 + c);
            const std::uint16_t p6 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(r) * input_rows + 16 + 8 + c);
            const std::uint16_t p7 = *reinterpret_cast<const std::uint16_t*>(
                src + static_cast<std::int64_t>(8 + r) * input_rows + 16 + 8 + c);
            const uint4 packed = make_uint4(
                static_cast<unsigned>(p0) | (static_cast<unsigned>(p1) << 16),
                static_cast<unsigned>(p2) | (static_cast<unsigned>(p3) << 16),
                static_cast<unsigned>(p4) | (static_cast<unsigned>(p5) << 16),
                static_cast<unsigned>(p6) | (static_cast<unsigned>(p7) << 16));
            store_vec(reinterpret_cast<uint4*>(
                          codes_at(half * kSmallTileCodeBytes + unit * 512 + st_lane * 16)),
                      packed);
        }
    };

    float gate_accum[kSmallTileNCols][4] = {};
    float up_accum[kSmallTileNCols][4] = {};
    unsigned gate_partial[kSmallTileNCols][2] = {};
    unsigned up_partial[kSmallTileNCols][2] = {};

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    for (int tile = 0; tile < k_tiles; ++tile) {
        stage_activation(tile);
        stage_codes(tile);
        __syncthreads();
        const __half2 gate_scale = fp8_block_stage_scale<BroadcastScale>(
            scales, static_cast<std::int64_t>(row0 / 128) * scale_tiles + tile / 2, lane);
        const __half2 up_scale = fp8_block_stage_scale<BroadcastScale>(
            scales, static_cast<std::int64_t>((row0 + intermediate_rows) / 128) * scale_tiles +
                        tile / 2,
            lane);
#pragma unroll
        for (int k_pair = 0; k_pair < 2; ++k_pair) {
            unsigned b_fragments[2][kSmallTileNCols][2];
#pragma unroll
            for (int step = 0; step < 2; ++step) {
                const int k_step = k_pair * 32 + step * 16;
#pragma unroll
                for (int ni = 0; ni < kSmallTileNCols; ++ni) {
                    const int row = ni * 8 + b_inner_row;
                    const int column = k_step + b_k_offset;
                    ldmatrix_x2(b_fragments[step][ni][0], b_fragments[step][ni][1],
                                smem_addr(activations_at(row * kHmmaBlockK +
                                                          fp8_block_swizzle(row, column))));
                }
            }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                unsigned (&partial)[kSmallTileNCols][2] =
                    (half == 0) ? gate_partial : up_partial;
                const __half2 scale = (half == 0) ? gate_scale : up_scale;
                const int unit = warp * 2 + k_pair;
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    codes_at(half * kSmallTileCodeBytes + unit * 512 + lane * 16));
                unsigned a_unit[8];
                fp8_dequant_a_unit(fp8_code_table, raw, scale, a_unit);
#pragma unroll
                for (int step = 0; step < 2; ++step) {
#pragma unroll
                    for (int ni = 0; ni < kSmallTileNCols; ++ni) {
                        fp8_mma_f16_f16acc(partial[ni], a_unit[step * 4], a_unit[step * 4 + 1],
                                           a_unit[step * 4 + 2], a_unit[step * 4 + 3],
                                           b_fragments[step][ni][0], b_fragments[step][ni][1]);
                    }
                }
            }
        }
#pragma unroll
        for (int ni = 0; ni < kSmallTileNCols; ++ni) {
            fp8_fold_f16_partial(gate_accum[ni], gate_partial[ni]);
            fp8_fold_f16_partial(up_accum[ni], up_partial[ni]);
        }
        __syncthreads();
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
    const int row_a = row0 + warp * kSmallTileWarpRows + mma_row;
    const int row_b = row_a + 8;
#pragma unroll
    for (int ni = 0; ni < kSmallTileNCols; ++ni) {
        const int col_a = col0 + ni * 8 + mma_col;
        const int col_b = col_a + 1;
        const float* gate = gate_accum[ni];
        const float* up = up_accum[ni];
        if (row_a < intermediate_rows) {
            if (col_a < tokens) {
                out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_a] =
                    __float2bfloat16_rn(silu(gate[0]) * up[0]);
            }
            if (col_b < tokens) {
                out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_a] =
                    __float2bfloat16_rn(silu(gate[1]) * up[1]);
            }
        }
        if (row_b < intermediate_rows) {
            if (col_a < tokens) {
                out[static_cast<std::int64_t>(col_a) * intermediate_rows + row_b] =
                    __float2bfloat16_rn(silu(gate[2]) * up[2]);
            }
            if (col_b < tokens) {
                out[static_cast<std::int64_t>(col_b) * intermediate_rows + row_b] =
                    __float2bfloat16_rn(silu(gate[3]) * up[3]);
            }
        }
    }
}

// Shared-broadcast register-dequantized linear HMMA (milestone-2 scheme C applied to the
// single-GEMM linear leaf). This is the linear-leaf analogue of the SwiGLU broadcast kernel:
// the raw row-major codes of one K tile are reordered into the packed 16x32 shared layout once
// per block (block-level, so weight DRAM traffic stays 1x), then every warp re-reads its own
// 64-row band from shared with LDS.128 and decodes it in registers straight into mma A fragments.
// It drops the STS-of-decoded + ldmatrix weight path of the staged kernel, replacing it with a
// smaller raw-code STS and per-warp register decode; the activation B fragments keep the shared
// staging path. The linear leaf's 128x128 tile has the same 2x4 warp grid (warp-col redundancy 4x)
// as the staged kernel, so this reuses the 16x32 packed layout with 8 row tiles instead of the
// SwiGLU kernel's 4. Routed by NINFER_FP8_BLOCK_LINEAR_BROADCAST while under development.
constexpr int kLinearBroadcastCodeBytes =
    kHmmaBlockRows * kHmmaBlockK * static_cast<int>(sizeof(std::uint8_t));
constexpr int kLinearBroadcastSharedBytes =
    kLinearBroadcastCodeBytes + kHmmaBlockCols * kHmmaBlockK * static_cast<int>(sizeof(__half));
static_assert(kLinearBroadcastCodeBytes == 8192);
static_assert(kLinearBroadcastSharedBytes == 24576);

template <class Output, bool BroadcastScale>
__global__ __launch_bounds__(kHmmaThreads) void fp8_block_linear_hmma_broadcast_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, Output output, int output_rows, int input_rows,
    int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    auto codes_at = [&](int offset) -> std::uint8_t* {
        return reinterpret_cast<std::uint8_t*>(shared) + offset;
    };
    auto activations_at = [&](int offset) -> __half* {
        return shared + kLinearBroadcastCodeBytes / static_cast<int>(sizeof(__half)) + offset;
    };

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp / (kHmmaBlockCols / kHmmaWarpCols);
    const int warp_col = warp % (kHmmaBlockCols / kHmmaWarpCols);
    const int row0 = static_cast<int>(blockIdx.x) * kHmmaBlockRows;
    const int col0 = static_cast<int>(blockIdx.y) * kHmmaBlockCols;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int scale_tiles = input_rows / 128;
    const int k_tiles = input_rows / kHmmaBlockK;

    const __nv_bfloat16* activation_sources[kHmmaItems];
    int activation_destinations[kHmmaItems];
#pragma unroll
    for (int i = 0; i < kHmmaItems; ++i) {
        const int item = tid + i * kHmmaThreads;
        const int row = item / (kHmmaBlockK / 8);
        const int k8 = item - row * (kHmmaBlockK / 8);
        const int token = col0 + row;
        activation_sources[i] = x + static_cast<std::int64_t>(token) * input_rows + k8 * 8;
        activation_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
    }

    auto stage_activation = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kHmmaItems; ++i) {
            const int token = col0 + (tid + i * kHmmaThreads) / (kHmmaBlockK / 8);
            const int4 packed = token < tokens
                                    ? load_ldg<int4>(activation_sources[i] + k0)
                                    : make_int4(0, 0, 0, 0);
            stage_bf16_word(activations_at(activation_destinations[i]), packed);
        }
    };

    // Stage the packed 16x32-layout codes of this K tile into shared. The persistent layout
    // already orders each lane's 16 codes contiguously, so this is a straight LDG.128 + STS.128
    // per unit (no row-major reshuffle), matching the external Marlin's cp_async4 (which degrades
    // to the same synchronous load+store on sm_75).
    auto stage_codes = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
        const int units_per_row = input_rows / 32;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int lu = tid + i * kHmmaThreads;
            const int unit = lu >> 5;
            const int st_lane = lu & 31;
            const int unit_row = row0 + (unit >> 1) * 16;
            const int unit_k = k0 + (unit & 1) * 32;
            const int unit_index = (unit_row / 16) * units_per_row + (unit_k / 32);
            const uint4 raw = load_ldg<uint4>(
                reinterpret_cast<const uint4*>(codes) + unit_index * 32 + st_lane);
            store_vec(reinterpret_cast<uint4*>(
                          codes_at(unit * 512 + st_lane * 16)),
                      raw);
        }
    };

    float accum[kHmmaRows][kHmmaCols][4] = {};
    unsigned accum_h[kHmmaRows][kHmmaCols][2] = {};

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    for (int tile = 0; tile < k_tiles; ++tile) {
        stage_activation(tile);
        stage_codes(tile);
        __syncthreads();
        const __half2 scale = fp8_block_stage_scale_bias8(
            scales, static_cast<std::int64_t>(row0 / 128) * scale_tiles + tile / 2, lane);
#pragma unroll
        for (int k_pair = 0; k_pair < 2; ++k_pair) {
            unsigned b_fragments[2][kHmmaCols][2];
#pragma unroll
            for (int step = 0; step < 2; ++step) {
                const int k_step = k_pair * 32 + step * 16;
#pragma unroll
                for (int ni = 0; ni < kHmmaCols; ++ni) {
                    const int row = warp_col * kHmmaWarpCols + ni * 8 + b_inner_row;
                    const int column = k_step + b_k_offset;
                    ldmatrix_x2(b_fragments[step][ni][0], b_fragments[step][ni][1],
                                smem_addr(activations_at(row * kHmmaBlockK +
                                                          fp8_block_swizzle(row, column))));
                }
            }
#pragma unroll
            for (int mi = 0; mi < kHmmaRows; ++mi) {
                const int unit = (warp_row * 4 + mi) * 2 + k_pair;
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    codes_at(unit * 512 + lane * 16));
                unsigned a_unit[8];
                fp8_dequant_a_unit_alu(raw, scale, a_unit);
#pragma unroll
                for (int step = 0; step < 2; ++step) {
#pragma unroll
                    for (int ni = 0; ni < kHmmaCols; ++ni) {
                        fp8_mma_f16_f16acc(accum_h[mi][ni], a_unit[step * 4],
                                           a_unit[step * 4 + 1], a_unit[step * 4 + 2],
                                           a_unit[step * 4 + 3], b_fragments[step][ni][0],
                                           b_fragments[step][ni][1]);
                    }
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < kHmmaRows; ++mi) {
#pragma unroll
            for (int ni = 0; ni < kHmmaCols; ++ni) {
                fp8_fold_f16_partial(accum[mi][ni], accum_h[mi][ni]);
            }
        }
        __syncthreads();
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kHmmaRows; ++mi) {
        const int row_a = row0 + warp_row * kHmmaWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kHmmaCols; ++ni) {
            const int col_a = col0 + warp_col * kHmmaWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            const float* values = accum[mi][ni];
            if (row_a < output_rows) {
                if (col_a < tokens) { output.store(row_a, col_a, values[0]); }
                if (col_b < tokens) { output.store(row_a, col_b, values[1]); }
            }
            if (row_b < output_rows) {
                if (col_a < tokens) { output.store(row_b, col_a, values[2]); }
                if (col_b < tokens) { output.store(row_b, col_b, values[3]); }
            }
        }
    }
}

// 64-token x 256-N packed broadcast kernel: the external Marlin's tile orientation (narrow token
// band, wide N band). The 8-warp grid is 4 token bands x 2 N bands, so each warp owns 16 tokens x
// 128 rows; the activation footprint halves relative to the 128-token tile while the weight plane
// doubles, moving shared-traffic redundancy from the LSU side to the DRAM side. The weight path is
// the packed 16x32 layout staged block-level (LDG.128 + STS.128) and decoded in registers with the
// ALU bit transform; the activation keeps the shared staging path. Routed by
// NINFER_FP8_BLOCK_LINEAR_PACKED256 while under development.
constexpr int kP256BlockRows = 256;
constexpr int kP256BlockCols = 64;
constexpr int kP256WarpRows = 128;
constexpr int kP256WarpCols = 16;
constexpr int kP256Mi = kP256WarpRows / 16;
constexpr int kP256Ni = kP256WarpCols / 8;
constexpr int kP256ActItems = kP256BlockCols * (kHmmaBlockK / 8) / kHmmaThreads;
constexpr int kP256CodeBytes = kP256BlockRows * kHmmaBlockK;
constexpr int kP256SharedBytes =
    kP256CodeBytes + kP256BlockCols * kHmmaBlockK * static_cast<int>(sizeof(__half));
static_assert((kP256BlockRows / kP256WarpRows) * (kP256BlockCols / kP256WarpCols) == kHmmaWarps);
static_assert(kP256ActItems * kHmmaThreads == kP256BlockCols * (kHmmaBlockK / 8));
static_assert(kP256SharedBytes + 1024 <= 64 * 1024);

template <class Output, bool NoFold = false>
__global__ __launch_bounds__(kHmmaThreads) void fp8_block_linear_hmma_packed256_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, Output output, int output_rows, int input_rows,
    int tokens) {
    extern __shared__ __align__(16) __half shared[];
    auto codes_at = [&](int offset) -> std::uint8_t* {
        return reinterpret_cast<std::uint8_t*>(shared) + offset;
    };
    auto activations_at = [&](int offset) -> __half* {
        return shared + kP256CodeBytes / static_cast<int>(sizeof(__half)) + offset;
    };

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp / (kP256BlockCols / kP256WarpCols);  // warp / 4 -> N band
    const int warp_col = warp % (kP256BlockCols / kP256WarpCols);  // warp % 4 -> token band
    const int row0 = static_cast<int>(blockIdx.x) * kP256BlockRows;
    const int col0 = static_cast<int>(blockIdx.y) * kP256BlockCols;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int scale_tiles = input_rows / 128;
    const int k_tiles = input_rows / kHmmaBlockK;
    const int units_per_row = input_rows / 32;

    const __nv_bfloat16* activation_sources[kP256ActItems];
    int activation_destinations[kP256ActItems];
#pragma unroll
    for (int i = 0; i < kP256ActItems; ++i) {
        const int item = tid + i * kHmmaThreads;
        const int row = item / (kHmmaBlockK / 8);
        const int k8 = item - row * (kHmmaBlockK / 8);
        const int token = col0 + row;
        activation_sources[i] = x + static_cast<std::int64_t>(token) * input_rows + k8 * 8;
        activation_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
    }

    auto stage_activation = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kP256ActItems; ++i) {
            const int token = col0 + (tid + i * kHmmaThreads) / (kHmmaBlockK / 8);
            const int4 packed = token < tokens
                                    ? load_ldg<int4>(activation_sources[i] + k0)
                                    : make_int4(0, 0, 0, 0);
            stage_bf16_word(activations_at(activation_destinations[i]), packed);
        }
    };

    auto stage_codes = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lu = tid + i * kHmmaThreads;  // 0..1023
            const int unit = lu >> 5;
            const int st_lane = lu & 31;
            const int unit_row = row0 + (unit >> 1) * 16;
            const int unit_k = k0 + (unit & 1) * 32;
            const int unit_index = (unit_row / 16) * units_per_row + (unit_k / 32);
            const uint4 raw = load_ldg<uint4>(
                reinterpret_cast<const uint4*>(codes) + unit_index * 32 + st_lane);
            store_vec(reinterpret_cast<uint4*>(codes_at(unit * 512 + st_lane * 16)), raw);
        }
    };

    float accum[kP256Mi][kP256Ni][4] = {};
    unsigned accum_h[kP256Mi][kP256Ni][2] = {};
    if constexpr (NoFold) {
        (void)accum;
    }

    for (int tile = 0; tile < k_tiles; ++tile) {
        stage_activation(tile);
        stage_codes(tile);
        __syncthreads();
        const __half2 scale = fp8_block_stage_scale_bias8(
            scales, static_cast<std::int64_t>(row0 / 128 + warp_row) * scale_tiles + tile / 2,
            lane);
#pragma unroll
        for (int k_pair = 0; k_pair < 2; ++k_pair) {
            unsigned b_fragments[2][kP256Ni][2];
#pragma unroll
            for (int step = 0; step < 2; ++step) {
                const int k_step = k_pair * 32 + step * 16;
#pragma unroll
                for (int ni = 0; ni < kP256Ni; ++ni) {
                    const int row = warp_col * kP256WarpCols + ni * 8 + b_inner_row;
                    const int column = k_step + b_k_offset;
                    ldmatrix_x2(b_fragments[step][ni][0], b_fragments[step][ni][1],
                                smem_addr(activations_at(row * kHmmaBlockK +
                                                          fp8_block_swizzle(row, column))));
                }
            }
#pragma unroll
            for (int mi = 0; mi < kP256Mi; ++mi) {
                const int unit = (warp_row * kP256Mi + mi) * 2 + k_pair;
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    codes_at(unit * 512 + lane * 16));
                unsigned a_unit[8];
                fp8_dequant_a_unit_alu(raw, scale, a_unit);
#pragma unroll
                for (int step = 0; step < 2; ++step) {
#pragma unroll
                    for (int ni = 0; ni < kP256Ni; ++ni) {
                        fp8_mma_f16_f16acc(accum_h[mi][ni], a_unit[step * 4],
                                           a_unit[step * 4 + 1], a_unit[step * 4 + 2],
                                           a_unit[step * 4 + 3], b_fragments[step][ni][0],
                                           b_fragments[step][ni][1]);
                    }
                }
            }
        }
        if constexpr (!NoFold) {
#pragma unroll
            for (int mi = 0; mi < kP256Mi; ++mi) {
#pragma unroll
                for (int ni = 0; ni < kP256Ni; ++ni) {
                    fp8_fold_f16_partial(accum[mi][ni], accum_h[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kP256Mi; ++mi) {
        const int row_a = row0 + warp_row * kP256WarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kP256Ni; ++ni) {
            const int col_a = col0 + warp_col * kP256WarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            float values[4];
            if constexpr (NoFold) {
                const float2 low =
                    __half22float2(*reinterpret_cast<const __half2*>(&accum_h[mi][ni][0]));
                const float2 high =
                    __half22float2(*reinterpret_cast<const __half2*>(&accum_h[mi][ni][1]));
                values[0] = low.x;
                values[1] = low.y;
                values[2] = high.x;
                values[3] = high.y;
            } else {
                const float* source = accum[mi][ni];
                values[0] = source[0];
                values[1] = source[1];
                values[2] = source[2];
                values[3] = source[3];
            }
            if (row_a < output_rows) {
                if (col_a < tokens) { output.store(row_a, col_a, values[0]); }
                if (col_b < tokens) { output.store(row_a, col_b, values[1]); }
            }
            if (row_b < output_rows) {
                if (col_a < tokens) { output.store(row_b, col_a, values[2]); }
                if (col_b < tokens) { output.store(row_b, col_b, values[3]); }
            }
        }
    }
}

// Packed-layout (16x32 unit) register-dequantized linear HMMA, milestone-3 (scheme-C complement):
// the weight codes live in a persistent 16x32 layout where each lane's 16 codes for two consecutive
// k-steps are contiguous, so one LDG.128 per lane per 16-row unit feeds two k-steps of mma with no
// shared staging and no weight ldmatrix. The tile is 128 rows x 128 tokens with a 2x4 warp grid
// (2 row bands x 4 token bands): the weight LDG redundancy moves to the DRAM side (which is not the
// prefill bottleneck) while the activation LSU stays at the 2x minimum of the staged kernel's grid.
// This mirrors the external Marlin's redundancy profile (weight LDG x4, activation LSU x2). The
// activation B fragments keep the shared staging path. Routed by NINFER_FP8_BLOCK_LINEAR_PACKED
// while under development; the converter must write the 16x32 persistent layout first.
constexpr int kPackedBlockRows = 128;
constexpr int kPackedBlockCols = 128;
constexpr int kPackedWarpRows = 64;
constexpr int kPackedWarpCols = 32;
constexpr int kPackedMi = kPackedWarpRows / 16;
constexpr int kPackedNi = kPackedWarpCols / 8;
constexpr int kPackedActItems = kPackedBlockCols * (kHmmaBlockK / 8) / kHmmaThreads;
constexpr int kPackedSharedBytes =
    kPackedBlockCols * kHmmaBlockK * static_cast<int>(sizeof(__half));
static_assert((kPackedBlockRows / kPackedWarpRows) * (kPackedBlockCols / kPackedWarpCols) ==
              kHmmaWarps);
static_assert(kPackedActItems * kHmmaThreads == kPackedBlockCols * (kHmmaBlockK / 8));
static_assert(kPackedSharedBytes + 1024 <= 64 * 1024);

template <class Output, bool BroadcastScale>
__global__ __launch_bounds__(kHmmaThreads) void fp8_block_linear_hmma_packed_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, Output output, int output_rows, int input_rows,
    int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    auto activations_at = [&](int offset) -> __half* { return shared + offset; };

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp / (kPackedBlockCols / kPackedWarpCols);
    const int warp_col = warp % (kPackedBlockCols / kPackedWarpCols);
    const int row0 = static_cast<int>(blockIdx.x) * kPackedBlockRows;
    const int col0 = static_cast<int>(blockIdx.y) * kPackedBlockCols;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int scale_tiles = input_rows / 128;
    const int k_tiles = input_rows / kHmmaBlockK;
    const int units_per_row = input_rows / 32;

    const __nv_bfloat16* activation_sources[kPackedActItems];
    int activation_destinations[kPackedActItems];
#pragma unroll
    for (int i = 0; i < kPackedActItems; ++i) {
        const int item = tid + i * kHmmaThreads;
        const int row = item / (kHmmaBlockK / 8);
        const int k8 = item - row * (kHmmaBlockK / 8);
        const int token = col0 + row;
        activation_sources[i] = x + static_cast<std::int64_t>(token) * input_rows + k8 * 8;
        activation_destinations[i] = row * kHmmaBlockK + fp8_block_swizzle(row, k8 * 8);
    }

    auto stage_activation = [&](int tile) {
        const int k0 = tile * kHmmaBlockK;
#pragma unroll
        for (int i = 0; i < kPackedActItems; ++i) {
            const int token = col0 + (tid + i * kHmmaThreads) / (kHmmaBlockK / 8);
            const int4 packed = token < tokens
                                    ? load_ldg<int4>(activation_sources[i] + k0)
                                    : make_int4(0, 0, 0, 0);
            stage_bf16_word(activations_at(activation_destinations[i]), packed);
        }
    };

    float accum[kPackedMi][kPackedNi][4] = {};
    unsigned accum_h[kPackedMi][kPackedNi][2] = {};

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    for (int tile = 0; tile < k_tiles; ++tile) {
        stage_activation(tile);
        __syncthreads();
        const int k0 = tile * kHmmaBlockK;
        const __half2 scale = fp8_block_stage_scale_bias8(
            scales, static_cast<std::int64_t>(row0 / 128) * scale_tiles + tile / 2, lane);
#pragma unroll
        for (int k_pair = 0; k_pair < 2; ++k_pair) {
            const int unit_k = k0 + k_pair * 32;
            unsigned b_fragments[2][kPackedNi][2];
#pragma unroll
            for (int step = 0; step < 2; ++step) {
                const int k_step = k_pair * 32 + step * 16;
#pragma unroll
                for (int ni = 0; ni < kPackedNi; ++ni) {
                    const int row = warp_col * kPackedWarpCols + ni * 8 + b_inner_row;
                    const int column = k_step + b_k_offset;
                    ldmatrix_x2(b_fragments[step][ni][0], b_fragments[step][ni][1],
                                smem_addr(activations_at(row * kHmmaBlockK +
                                                          fp8_block_swizzle(row, column))));
                }
            }
#pragma unroll
            for (int mi = 0; mi < kPackedMi; ++mi) {
                const int unit_row = row0 + warp_row * kPackedWarpRows + mi * 16;
                const int unit_index = (unit_row / 16) * units_per_row + (unit_k / 32);
                const uint4 raw = load_ldg<uint4>(
                    reinterpret_cast<const uint4*>(codes) + unit_index * 32 + lane);
                unsigned a_unit[8];
                fp8_dequant_a_unit_alu(raw, scale, a_unit);
#pragma unroll
                for (int step = 0; step < 2; ++step) {
#pragma unroll
                    for (int ni = 0; ni < kPackedNi; ++ni) {
                        fp8_mma_f16_f16acc(accum_h[mi][ni], a_unit[step * 4],
                                           a_unit[step * 4 + 1], a_unit[step * 4 + 2],
                                           a_unit[step * 4 + 3], b_fragments[step][ni][0],
                                           b_fragments[step][ni][1]);
                    }
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < kPackedMi; ++mi) {
#pragma unroll
            for (int ni = 0; ni < kPackedNi; ++ni) {
                fp8_fold_f16_partial(accum[mi][ni], accum_h[mi][ni]);
            }
        }
        __syncthreads();
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kPackedMi; ++mi) {
        const int row_a = row0 + warp_row * kPackedWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kPackedNi; ++ni) {
            const int col_a = col0 + warp_col * kPackedWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            const float* values = accum[mi][ni];
            if (row_a < output_rows) {
                if (col_a < tokens) { output.store(row_a, col_a, values[0]); }
                if (col_b < tokens) { output.store(row_a, col_b, values[1]); }
            }
            if (row_b < output_rows) {
                if (col_a < tokens) { output.store(row_b, col_a, values[2]); }
                if (col_b < tokens) { output.store(row_b, col_b, values[3]); }
            }
        }
    }
}

template <bool Marlin, int TokensPerBlock>
__global__ __launch_bounds__(kThreads) void fp8_block_linear_swiglu_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int intermediate_rows, int input_rows, int tokens) {
    constexpr int kTokensPerBlock = TokensPerBlock;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    const int token_begin = static_cast<int>(blockIdx.y) * kTokensPerBlock;
    const int k_tiles = input_rows / 128;
    __shared__ float fp8_code_table[kFp8CodeCount];
    build_fp8_code_table_float(fp8_code_table, static_cast<int>(threadIdx.x), kThreads);
    __syncthreads();
    const int up_row = row + intermediate_rows;
    const int total_rows = 2 * intermediate_rows;
    float gate_totals[kTokensPerBlock] = {};
    float up_totals[kTokensPerBlock] = {};

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int k_begin = k_tile * 128 + lane * 4;
        const std::uint32_t packed_gate =
            Marlin ? marlin_fp8_block_word(codes, row, k_begin, total_rows)
                   : *reinterpret_cast<const std::uint32_t*>(
                         codes + static_cast<std::int64_t>(row) * input_rows + k_begin);
        const std::uint32_t packed_up =
            Marlin ? marlin_fp8_block_word(codes, up_row, k_begin, total_rows)
                   : *reinterpret_cast<const std::uint32_t*>(
                         codes + static_cast<std::int64_t>(up_row) * input_rows + k_begin);
        const float2 gate01 = decode_fp8_pair(fp8_code_table,
                                              static_cast<std::uint16_t>(packed_gate));
        const float2 gate23 = decode_fp8_pair(fp8_code_table,
                                              static_cast<std::uint16_t>(packed_gate >> 16));
        const float2 up01 = decode_fp8_pair(fp8_code_table,
                                            static_cast<std::uint16_t>(packed_up));
        const float2 up23 = decode_fp8_pair(fp8_code_table,
                                            static_cast<std::uint16_t>(packed_up >> 16));
        const float gate_scale =
            __bfloat162float(scales[static_cast<std::int64_t>(row / 128) * k_tiles + k_tile]);
        const float up_scale =
            __bfloat162float(scales[static_cast<std::int64_t>(up_row / 128) * k_tiles + k_tile]);
        const __nv_bfloat16* activation =
            x + static_cast<std::int64_t>(token_begin) * input_rows + k_begin;
#pragma unroll
        for (int token_slot = 0; token_slot < kTokensPerBlock; ++token_slot) {
            const int token = token_begin + token_slot;
            if (token < tokens) {
                const uint2 packed_activation =
                    load_ldg<uint2>(reinterpret_cast<const uint2*>(activation));
                const float2 activation01 = bf16x2_wide(packed_activation.x);
                const float2 activation23 = bf16x2_wide(packed_activation.y);
                float gate_partial = 0.0F;
                float up_partial = 0.0F;
                gate_partial = fmaf(gate01.x, activation01.x, gate_partial);
                gate_partial = fmaf(gate01.y, activation01.y, gate_partial);
                gate_partial = fmaf(gate23.x, activation23.x, gate_partial);
                gate_partial = fmaf(gate23.y, activation23.y, gate_partial);
                up_partial = fmaf(up01.x, activation01.x, up_partial);
                up_partial = fmaf(up01.y, activation01.y, up_partial);
                up_partial = fmaf(up23.x, activation23.x, up_partial);
                up_partial = fmaf(up23.y, activation23.y, up_partial);
                gate_totals[token_slot] = fmaf(gate_scale, gate_partial, gate_totals[token_slot]);
                up_totals[token_slot] = fmaf(up_scale, up_partial, up_totals[token_slot]);
            }
            activation += static_cast<std::int64_t>(input_rows);
        }
    }

#pragma unroll
    for (int token_slot = 0; token_slot < kTokensPerBlock; ++token_slot) {
        const int token = token_begin + token_slot;
        if (token < tokens) {
            const float gate = warp_sum(gate_totals[token_slot]);
            const float up = warp_sum(up_totals[token_slot]);
            if (lane == 0) {
                out[static_cast<std::int64_t>(token) * intermediate_rows + row] =
                    __float2bfloat16_rn(silu(gate) * up);
            }
        }
    }
}

// --- Marlin persistent-layout fused SwiGLU ------------------------------------------------
//
// One CTA covers 64 intermediate rows (the gate and up halves share the same output rows) and
// 128 tokens, with 256 threads (8 warps of 32 rows x 32 tokens) so a single accumulator set
// stays small while each warp's ldmatrix reads are amortised over twice the mma of the 16-row
// tiling. The single shared buffer holds both weight halves and the activation tile: the next K
// tile's global loads are issued before the current tile's staging, and the staging is performed
// after the current tile's mma, so one tile's ldg / decode / STS work runs under the other's
// tensor work. The 64-wide swizzle is shared with the projection leaf. The 256-thread shape is
// what makes the 32-row warp tile affordable: at 512 threads the register file caps the thread
// at 128 registers, and an 8-mma-tile accumulator set does not fit beside the staging state.
constexpr int kMsBlockRows = 64;
constexpr int kMsBlockCols = 128;
constexpr int kMsBlockK = 64;
constexpr int kMsWarpRows = 32;
constexpr int kMsWarpCols = 32;
constexpr int kMsMi = kMsWarpRows / 16;
constexpr int kMsNi = kMsWarpCols / 8;
constexpr int kMsThreads = 256;
constexpr int kMsWarps = kMsThreads / 32;
constexpr int kMsActivationItems = kMsBlockCols * (kMsBlockK / 8) / kMsThreads;
constexpr int kMsWeightItems = 2 * kMsBlockRows * (kMsBlockK / 32) * 2 / kMsThreads;
constexpr int kMsSharedBytes =
    (2 * kMsBlockRows + kMsBlockCols) * kMsBlockK * static_cast<int>(sizeof(__half));

static_assert(kMsWarps == (kMsBlockRows / kMsWarpRows) * (kMsBlockCols / kMsWarpCols));
static_assert(kMsWeightItems * kMsThreads == 2 * kMsBlockRows * (kMsBlockK / 32) * 2);
static_assert(kMsActivationItems * kMsThreads == kMsBlockCols * (kMsBlockK / 8));
static_assert(kMsSharedBytes + 1024 <= 64 * 1024);

__device__ __forceinline__ uint2 decode_marlin_quad(const unsigned* table, std::uint32_t word,
                                                    __half2 scale) {
    const __half2 low =
        __hmul2(decode_fp8_pair_half(table, static_cast<std::uint16_t>(word)), scale);
    const __half2 high =
        __hmul2(decode_fp8_pair_half(table, static_cast<std::uint16_t>(word >> 16)), scale);
    uint2 packed;
    packed.x = *reinterpret_cast<const unsigned*>(&low);
    packed.y = *reinterpret_cast<const unsigned*>(&high);
    return packed;
}

__global__ __launch_bounds__(kMsThreads, 1) void marlin_fp8_block_swiglu_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ out, int half_rows,
    int total_rows, int input_rows, int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    __half* weights = shared;
    __half* activations = shared + 2 * kMsBlockRows * kMsBlockK;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp >> 2;
    const int warp_col = warp & 3;
    const int row0 = static_cast<int>(blockIdx.x) * kMsBlockRows;
    const int col0 = static_cast<int>(blockIdx.y) * kMsBlockCols;
    const int k_tiles = input_rows / kMsBlockK;
    const int k_groups = input_rows / 128;
    const int n_tiles = total_rows / 32;

    const int a_matrix = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;

    int activation_destinations[kMsActivationItems];
    const __nv_bfloat16* activation_sources[kMsActivationItems];
#pragma unroll
    for (int i = 0; i < kMsActivationItems; ++i) {
        const int item = tid + i * kMsThreads;
        const int row = item / (kMsBlockK / 8);
        const int k8 = item - row * (kMsBlockK / 8);
        activation_destinations[i] = row * kMsBlockK + fp8_block_swizzle(row, k8 * 8);
        activation_sources[i] = x + static_cast<std::int64_t>(col0 + row) * input_rows + k8 * 8;
    }

    // One aligned 16-code unit per thread per K tile: the tile's 8 x 64 units (two weight
    // halves x two 32-row tiles x two 32-wide K halves) are handed out thread-major, so a warp
    // copies 512 contiguous bytes of one 32x32 code tile.
    int weight_destinations[kMsWeightItems][4];
    const std::uint8_t* weight_sources[kMsWeightItems];
    int scale_rows[kMsWeightItems];
#pragma unroll
    for (int i = 0; i < kMsWeightItems; ++i) {
        const int unit_slot = tid + i * kMsThreads;
        const int unit_tile = unit_slot >> 6;
        const int unit_in_tile = unit_slot & 63;
        const int unit_k_half = unit_tile >> 2;
        const int unit_half = (unit_tile >> 1) & 1;
        const int unit_n_tile = unit_tile & 1;
        // unit_in_tile -> (unit_t, unit_w) is a free relabelling: the load offset
        // 32*unit_t + 8*unit_w and all four destinations are derived from the same pair, so
        // permuting it only changes which thread copies which unit. This permutation swaps input
        // bits 2 and 4 and moves bit 3 into unit_w. It keeps a warp's 32 loads inside one
        // contiguous 512-byte window (so the global load coalescing is unchanged) while making the
        // four 8-byte staging stores of an eight-lane group cover four distinct swizzle quads with
        // both half-offsets each, which is conflict-free. Checked with a bank model calibrated
        // against ncu: the staging stores drop from 645 to 384 shared wavefronts per CTA K tile
        // and measure 389.
        const int unit_t = (unit_in_tile & 3) | (((unit_in_tile >> 4) & 1) << 2) |
                           (((unit_in_tile >> 2) & 1) << 3) | (((unit_in_tile >> 5) & 1) << 4);
        const int unit_w = ((unit_in_tile >> 3) & 1) * 2;
        const int unit_row = unit_half * kMsBlockRows + unit_n_tile * 32 + (unit_w >> 1) * 16 +
                             (unit_t >> 2);
        const int unit_column = unit_k_half * 32 + (unit_t & 3) * 4;
        weight_destinations[i][0] = unit_row * kMsBlockK + fp8_block_swizzle(unit_row, unit_column);
        weight_destinations[i][1] =
            unit_row * kMsBlockK + fp8_block_swizzle(unit_row, unit_column + 16);
        weight_destinations[i][2] =
            (unit_row + 8) * kMsBlockK + fp8_block_swizzle(unit_row + 8, unit_column);
        weight_destinations[i][3] =
            (unit_row + 8) * kMsBlockK + fp8_block_swizzle(unit_row + 8, unit_column + 16);
        weight_sources[i] =
            codes + static_cast<std::int64_t>(unit_k_half * n_tiles +
                                              unit_half * (half_rows / 32) + row0 / 32 +
                                              unit_n_tile) * 1024 +
            32 * unit_t + 8 * unit_w;
        scale_rows[i] = unit_half == 0 ? row0 : half_rows + row0;
    }
    const std::int64_t weight_source_stride = static_cast<std::int64_t>(2) * n_tiles * 1024;

    float gate_accum[kMsMi][kMsNi][4] = {};
    float up_accum[kMsMi][kMsNi][4] = {};
    unsigned gate_partial[kMsMi][kMsNi][2] = {};
    unsigned up_partial[kMsMi][kMsNi][2] = {};

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    // The staging loads for the next K tile are issued before the current tile's mma and staged
    // after it, so one tile's global latency and its copy/decode instructions run under the
    // other's tensor work.
    auto load_tile = [&](int tile, uint4 (&unit)[kMsWeightItems],
                         int4 (&packed)[kMsActivationItems],
                         __half2 (&scale)[kMsWeightItems]) {
#pragma unroll
        for (int i = 0; i < kMsWeightItems; ++i) {
            scale[i] = __half2half2(__float2half_rn(__bfloat162float(
                scales[fp8_block_scale_index(scale_rows[i], tile >> 1, k_groups)])));
            unit[i] = load_ldg<uint4>(
                reinterpret_cast<const uint4*>(weight_sources[i] + tile * weight_source_stride));
        }
#pragma unroll
        for (int i = 0; i < kMsActivationItems; ++i) {
            const int token = col0 + (tid + i * kMsThreads) / (kMsBlockK / 8);
            packed[i] = token < tokens ? load_ldg<int4>(activation_sources[i] + tile * kMsBlockK)
                                       : make_int4(0, 0, 0, 0);
        }
    };

    uint4 unit[kMsWeightItems];
    int4 packed[kMsActivationItems];
    __half2 half_scale[kMsWeightItems];
    load_tile(0, unit, packed, half_scale);

    for (int tile = 0; tile < k_tiles; ++tile) {
        const int next = tile + 1;
        uint4 next_unit[kMsWeightItems];
        int4 next_packed[kMsActivationItems];
        __half2 next_scale[kMsWeightItems];
        if (next < k_tiles) { load_tile(next, next_unit, next_packed, next_scale); }
        if (NINFER_FP8_BLOCK_ABLATION != 2) {
#pragma unroll
            for (int i = 0; i < kMsWeightItems; ++i) {
                store_vec(weights + weight_destinations[i][0],
                          decode_marlin_quad(fp8_code_table, unit[i].x, half_scale[i]));
                store_vec(weights + weight_destinations[i][1],
                          decode_marlin_quad(fp8_code_table, unit[i].y, half_scale[i]));
                store_vec(weights + weight_destinations[i][2],
                          decode_marlin_quad(fp8_code_table, unit[i].z, half_scale[i]));
                store_vec(weights + weight_destinations[i][3],
                          decode_marlin_quad(fp8_code_table, unit[i].w, half_scale[i]));
            }
#pragma unroll
            for (int i = 0; i < kMsActivationItems; ++i) {
                stage_bf16_word(activations + activation_destinations[i], packed[i]);
            }
        } else {
            // Keep the global loads live without touching shared memory.
            unsigned mixed = __float_as_uint(gate_accum[0][0][0]) ^
                             static_cast<unsigned>(packed[0].x) ^
                             static_cast<unsigned>(packed[0].w);
#pragma unroll
            for (int i = 0; i < kMsWeightItems; ++i) {
                mixed ^= unit[i].x ^ unit[i].y ^ unit[i].z ^ unit[i].w;
            }
            gate_accum[0][0][0] = __uint_as_float(mixed);
        }
        __syncthreads();

#pragma unroll
        for (int k_sub = 0; k_sub < kMsBlockK / 16; ++k_sub) {
            const int k_step = k_sub * 16;
            unsigned b_fragments[kMsNi][2];
#pragma unroll
            for (int ni = 0; ni < kMsNi; ++ni) {
                const int row = warp_col * kMsWarpCols + ni * 8 + b_inner_row;
                const int column = k_step + b_k_offset;
                if (NINFER_FP8_BLOCK_ABLATION == 3) {
                    b_fragments[ni][0] = static_cast<unsigned>(row + column);
                    b_fragments[ni][1] = static_cast<unsigned>(row * 31 + column);
                } else {
                    ldmatrix_x2(b_fragments[ni][0], b_fragments[ni][1],
                                smem_addr(activations + row * kMsBlockK +
                                          fp8_block_swizzle(row, column)));
                }
            }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int mi = 0; mi < kMsMi; ++mi) {
                    unsigned a_fragments[4];
                    const int row = half * kMsBlockRows + warp_row * kMsWarpRows + mi * 16 +
                                    a_row_offset;
                    const int column = k_step + a_col_offset;
                    if (NINFER_FP8_BLOCK_ABLATION == 3) {
                        a_fragments[0] = static_cast<unsigned>(row);
                        a_fragments[1] = static_cast<unsigned>(column);
                        a_fragments[2] = static_cast<unsigned>(row + column + half);
                        a_fragments[3] = static_cast<unsigned>(row ^ column);
                    } else {
                        ldmatrix_x4(a_fragments[0], a_fragments[1], a_fragments[2],
                                    a_fragments[3],
                                    smem_addr(weights + row * kMsBlockK +
                                              fp8_block_swizzle(row, column)));
                    }
                    unsigned (&partial)[kMsNi][2] =
                        (half == 0) ? gate_partial[mi] : up_partial[mi];
#pragma unroll
                    for (int ni = 0; ni < kMsNi; ++ni) {
                        if (NINFER_FP8_BLOCK_ABLATION == 1) {
                            partial[ni][0] ^=
                                a_fragments[0] ^ a_fragments[1] ^ b_fragments[ni][0];
                            partial[ni][1] ^=
                                a_fragments[2] ^ a_fragments[3] ^ b_fragments[ni][1];
                        } else {
                            fp8_mma_f16_f16acc(partial[ni], a_fragments[0], a_fragments[1],
                                               a_fragments[2], a_fragments[3], b_fragments[ni][0],
                                               b_fragments[ni][1]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < kMsMi; ++mi) {
#pragma unroll
            for (int ni = 0; ni < kMsNi; ++ni) {
                fp8_fold_f16_partial(gate_accum[mi][ni], gate_partial[mi][ni]);
                fp8_fold_f16_partial(up_accum[mi][ni], up_partial[mi][ni]);
            }
        }
        __syncthreads();
        if (next < k_tiles) {
#pragma unroll
            for (int i = 0; i < kMsWeightItems; ++i) {
                unit[i] = next_unit[i];
                half_scale[i] = next_scale[i];
            }
#pragma unroll
            for (int i = 0; i < kMsActivationItems; ++i) { packed[i] = next_packed[i]; }
        }
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kMsMi; ++mi) {
        const int row_a = row0 + warp_row * kMsWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kMsNi; ++ni) {
            const int col_a = col0 + warp_col * kMsWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            const float* gate = gate_accum[mi][ni];
            const float* up = up_accum[mi][ni];
            if (col_a < tokens) {
                out[static_cast<std::int64_t>(col_a) * half_rows + row_a] =
                    __float2bfloat16_rn(silu(gate[0]) * up[0]);
                out[static_cast<std::int64_t>(col_a) * half_rows + row_b] =
                    __float2bfloat16_rn(silu(gate[2]) * up[2]);
            }
            if (col_b < tokens) {
                out[static_cast<std::int64_t>(col_b) * half_rows + row_a] =
                    __float2bfloat16_rn(silu(gate[1]) * up[1]);
                out[static_cast<std::int64_t>(col_b) * half_rows + row_b] =
                    __float2bfloat16_rn(silu(gate[3]) * up[3]);
            }
        }
    }
}

template <class Output, bool Marlin, int TokensPerBlock>
void launch_scalar_tokens(const Tensor& x, const Weight& weight, Output output,
                          cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(weight.n / kRowsPerBlock),
                    static_cast<unsigned>((x.ne[1] + TokensPerBlock - 1) / TokensPerBlock));
    fp8_block_linear_kernel<Output, Marlin, TokensPerBlock><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output, weight.n, weight.k, x.ne[1]);
}

template <class Output, bool Marlin = false>
void launch(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    // The scalar kernel runs TokensPerBlock tokens per CTA, so its cost steps with the
    // token-block count ceil(T / TokensPerBlock). T=5..8 would need two 4-token blocks (weight
    // decode twice) at the 4-token default, so those widths use a matching token block instead:
    // one block, one weight decode, and the MTP4/MTP5 verify passes stay on the same cliff as T=4.
    if (x.ne[1] <= 4) {
        launch_scalar_tokens<Output, Marlin, 4>(x, weight, output, stream);
    } else if (x.ne[1] <= 5) {
        launch_scalar_tokens<Output, Marlin, 5>(x, weight, output, stream);
    } else if (x.ne[1] <= 6) {
        launch_scalar_tokens<Output, Marlin, 6>(x, weight, output, stream);
    } else if (x.ne[1] <= 7) {
        launch_scalar_tokens<Output, Marlin, 7>(x, weight, output, stream);
    } else {
        launch_scalar_tokens<Output, Marlin, 8>(x, weight, output, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Output, int Stages, bool BroadcastScale, bool NoFold, bool AluDecode>
void launch_hmma_staged(const Tensor& x, const Weight& weight, Output output,
                        cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(weight.n / kHmmaBlockRows),
                    static_cast<unsigned>((x.ne[1] + kHmmaBlockCols - 1) / kHmmaBlockCols));
    constexpr int kSharedBytes = hmma_shared_bytes<Stages>();
    ensure_func_attr_per_device(
        fp8_block_linear_hmma_kernel<Output, Stages, BroadcastScale, NoFold, AluDecode>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    fp8_block_linear_hmma_kernel<Output, Stages, BroadcastScale, NoFold, AluDecode>
        <<<grid, kHmmaThreads, kSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output, weight.n, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Output, int Stages, bool NoFold, bool AluDecode>
void launch_hmma_for_stage(const Tensor& x, const Weight& weight, Output output,
                           cudaStream_t stream) {
    if (fp8_block_hmma_scale_broadcast()) {
        launch_hmma_staged<Output, Stages, true, NoFold, AluDecode>(x, weight, output, stream);
    } else {
        launch_hmma_staged<Output, Stages, false, NoFold, AluDecode>(x, weight, output, stream);
    }
}

template <class Output, bool BroadcastScale>
void launch_hmma_broadcast(const Tensor& x, const Weight& weight, Output output,
                           cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(weight.n / kHmmaBlockRows),
                    static_cast<unsigned>((x.ne[1] + kHmmaBlockCols - 1) / kHmmaBlockCols));
    ensure_func_attr_per_device(
        fp8_block_linear_hmma_broadcast_kernel<Output, BroadcastScale>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kLinearBroadcastSharedBytes);
    fp8_block_linear_hmma_broadcast_kernel<Output, BroadcastScale>
        <<<grid, kHmmaThreads, kLinearBroadcastSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output, weight.n, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Output, bool BroadcastScale>
void launch_hmma_packed(const Tensor& x, const Weight& weight, Output output,
                        cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(weight.n / kPackedBlockRows),
                    static_cast<unsigned>((x.ne[1] + kPackedBlockCols - 1) / kPackedBlockCols));
    ensure_func_attr_per_device(
        fp8_block_linear_hmma_packed_kernel<Output, BroadcastScale>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kPackedSharedBytes);
    fp8_block_linear_hmma_packed_kernel<Output, BroadcastScale>
        <<<grid, kHmmaThreads, kPackedSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output, weight.n, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Output, bool NoFold>
void launch_hmma_packed256(const Tensor& x, const Weight& weight, Output output,
                           cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(weight.n / kP256BlockRows),
                    static_cast<unsigned>((x.ne[1] + kP256BlockCols - 1) / kP256BlockCols));
    ensure_func_attr_per_device(
        fp8_block_linear_hmma_packed256_kernel<Output, NoFold>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kP256SharedBytes);
    fp8_block_linear_hmma_packed256_kernel<Output, NoFold>
        <<<grid, kHmmaThreads, kP256SharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output, weight.n, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Output>
void launch_hmma(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    if (fp8_block_linear_packed256()) {
        if (fp8_block_nofold()) {
            launch_hmma_packed256<Output, true>(x, weight, output, stream);
        } else {
            launch_hmma_packed256<Output, false>(x, weight, output, stream);
        }
        return;
    }
    if (fp8_block_linear_packed()) {
        if (fp8_block_hmma_scale_broadcast()) {
            launch_hmma_packed<Output, true>(x, weight, output, stream);
        } else {
            launch_hmma_packed<Output, false>(x, weight, output, stream);
        }
        return;
    }
    if (fp8_block_linear_broadcast()) {
        if (fp8_block_hmma_scale_broadcast()) {
            launch_hmma_broadcast<Output, true>(x, weight, output, stream);
        } else {
            launch_hmma_broadcast<Output, false>(x, weight, output, stream);
        }
        return;
    }
    const bool nofold = fp8_block_nofold();
    const bool alu = fp8_block_alu_decode();
    if (fp8_block_hmma_stages() == 1) {
        if (nofold) {
            if (alu) {
                launch_hmma_for_stage<Output, 1, true, true>(x, weight, output, stream);
            } else {
                launch_hmma_for_stage<Output, 1, true, false>(x, weight, output, stream);
            }
        } else {
            if (alu) {
                launch_hmma_for_stage<Output, 1, false, true>(x, weight, output, stream);
            } else {
                launch_hmma_for_stage<Output, 1, false, false>(x, weight, output, stream);
            }
        }
    } else {
        if (nofold) {
            if (alu) {
                launch_hmma_for_stage<Output, 2, true, true>(x, weight, output, stream);
            } else {
                launch_hmma_for_stage<Output, 2, true, false>(x, weight, output, stream);
            }
        } else {
            if (alu) {
                launch_hmma_for_stage<Output, 2, false, true>(x, weight, output, stream);
            } else {
                launch_hmma_for_stage<Output, 2, false, false>(x, weight, output, stream);
            }
        }
    }
}

template <bool BroadcastScale, bool NoFold, bool AluDecode>
void launch_swiglu_hmma_staged(const Tensor& x, const Weight& weight, Tensor& out,
                               int intermediate_rows, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(intermediate_rows / kSwiGluRows),
                    static_cast<unsigned>((x.ne[1] + kSwiGluCols - 1) / kSwiGluCols));
    ensure_func_attr_per_device(fp8_block_swiglu_hmma_kernel<BroadcastScale, NoFold, AluDecode>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kSwiGluHmmaSharedBytes);
    fp8_block_swiglu_hmma_kernel<BroadcastScale, NoFold, AluDecode>
        <<<grid, kSwiGluThreads, kSwiGluHmmaSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), intermediate_rows, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <bool BroadcastScale>
void launch_swiglu_hmma_regdequant(const Tensor& x, const Weight& weight, Tensor& out,
                                   int intermediate_rows, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(intermediate_rows / kSwiGluRows),
                    static_cast<unsigned>((x.ne[1] + kSwiGluCols - 1) / kSwiGluCols));
    ensure_func_attr_per_device(fp8_block_swiglu_hmma_regdequant_kernel<BroadcastScale>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kSwiGluRegdequantSharedBytes);
    fp8_block_swiglu_hmma_regdequant_kernel<BroadcastScale>
        <<<grid, kSwiGluThreads, kSwiGluRegdequantSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), intermediate_rows, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <bool BroadcastScale>
void launch_swiglu_hmma_packed(const Tensor& x, const Weight& weight, Tensor& out,
                               int intermediate_rows, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(intermediate_rows / kSwiGluRows),
                    static_cast<unsigned>((x.ne[1] + kSwiGluCols - 1) / kSwiGluCols));
    ensure_func_attr_per_device(fp8_block_swiglu_hmma_packed_kernel<BroadcastScale>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kSwiGluRegdequantSharedBytes);
    fp8_block_swiglu_hmma_packed_kernel<BroadcastScale>
        <<<grid, kSwiGluThreads, kSwiGluRegdequantSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), intermediate_rows, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <bool BroadcastScale>
void launch_swiglu_hmma_broadcast(const Tensor& x, const Weight& weight, Tensor& out,
                                  int intermediate_rows, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(intermediate_rows / kSwiGluRows),
                    static_cast<unsigned>((x.ne[1] + kSwiGluCols - 1) / kSwiGluCols));
    ensure_func_attr_per_device(fp8_block_swiglu_hmma_broadcast_kernel<BroadcastScale>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kSwiGluBroadcastSharedBytes);
    fp8_block_swiglu_hmma_broadcast_kernel<BroadcastScale>
        <<<grid, kSwiGluThreads, kSwiGluBroadcastSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), intermediate_rows, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <bool BroadcastScale>
void launch_swiglu_hmma_smalltile(const Tensor& x, const Weight& weight, Tensor& out,
                                  int intermediate_rows, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(intermediate_rows / kSwiGluRows),
                    static_cast<unsigned>((x.ne[1] + kSmallTileCols - 1) / kSmallTileCols));
    ensure_func_attr_per_device(fp8_block_swiglu_hmma_smalltile_kernel<BroadcastScale>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kSmallTileSharedBytes);
    fp8_block_swiglu_hmma_smalltile_kernel<BroadcastScale>
        <<<grid, kSmallTileThreads, kSmallTileSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), intermediate_rows, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_swiglu_hmma(const Tensor& x, const Weight& weight, Tensor& out,
                        int intermediate_rows, cudaStream_t stream) {
    const bool broadcast = fp8_block_hmma_scale_broadcast();
    if (fp8_block_smalltile()) {
        if (broadcast) {
            launch_swiglu_hmma_smalltile<true>(x, weight, out, intermediate_rows, stream);
        } else {
            launch_swiglu_hmma_smalltile<false>(x, weight, out, intermediate_rows, stream);
        }
        return;
    }
    if (fp8_block_broadcast()) {
        if (broadcast) {
            launch_swiglu_hmma_broadcast<true>(x, weight, out, intermediate_rows, stream);
        } else {
            launch_swiglu_hmma_broadcast<false>(x, weight, out, intermediate_rows, stream);
        }
        return;
    }
    if (fp8_block_packed()) {
        if (broadcast) {
            launch_swiglu_hmma_packed<true>(x, weight, out, intermediate_rows, stream);
        } else {
            launch_swiglu_hmma_packed<false>(x, weight, out, intermediate_rows, stream);
        }
        return;
    }
    if (fp8_block_regdequant()) {
        if (broadcast) {
            launch_swiglu_hmma_regdequant<true>(x, weight, out, intermediate_rows, stream);
        } else {
            launch_swiglu_hmma_regdequant<false>(x, weight, out, intermediate_rows, stream);
        }
        return;
    }
    const bool nofold = fp8_block_nofold();
    const bool alu = fp8_block_alu_decode();
    if (broadcast) {
        if (nofold) {
            if (alu) {
                launch_swiglu_hmma_staged<true, true, true>(x, weight, out, intermediate_rows,
                                                            stream);
            } else {
                launch_swiglu_hmma_staged<true, true, false>(x, weight, out, intermediate_rows,
                                                             stream);
            }
        } else {
            if (alu) {
                launch_swiglu_hmma_staged<true, false, true>(x, weight, out, intermediate_rows,
                                                             stream);
            } else {
                launch_swiglu_hmma_staged<true, false, false>(x, weight, out, intermediate_rows,
                                                              stream);
            }
        }
    } else {
        if (nofold) {
            if (alu) {
                launch_swiglu_hmma_staged<false, true, true>(x, weight, out, intermediate_rows,
                                                             stream);
            } else {
                launch_swiglu_hmma_staged<false, true, false>(x, weight, out, intermediate_rows,
                                                              stream);
            }
        } else {
            if (alu) {
                launch_swiglu_hmma_staged<false, false, true>(x, weight, out, intermediate_rows,
                                                              stream);
            } else {
                launch_swiglu_hmma_staged<false, false, false>(x, weight, out, intermediate_rows,
                                                               stream);
            }
        }
    }
}

void launch_swiglu_marlin(const Tensor& x, const Weight& weight, Tensor& out,
                          int intermediate_rows, cudaStream_t stream) {
    if (std::getenv("NINFER_FP8_SHAPE_LOG") != nullptr) {
        std::fprintf(stderr, "swiglu n=%d k=%d tokens=%d inter=%d\n", weight.n, weight.k,
                     x.ne[1], intermediate_rows);
    }
    const dim3 grid(static_cast<unsigned>(intermediate_rows / kMsBlockRows),
                    static_cast<unsigned>((x.ne[1] + kMsBlockCols - 1) / kMsBlockCols));
    ensure_func_attr_per_device(marlin_fp8_block_swiglu_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, kMsSharedBytes);
    marlin_fp8_block_swiglu_kernel<<<grid, kMsThreads, kMsSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        static_cast<__nv_bfloat16*>(out.data), intermediate_rows, weight.n, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// --- Marlin persistent-layout projections -------------------------------------------------------
//
// One CTA covers 128 weight rows and 128 tokens with 256 threads (8 warps of 64 rows x 32 tokens).
// The 64-row warp tile halves the ldmatrix traffic per mma against a 32-row tiling because each
// warp's activation fragment load is amortised over twice the mma, and the 256-thread shape is
// what makes that tile affordable: at 512 threads the SM register file caps a thread at 128
// registers, which an accumulator set of this size does not fit beside the staging state. Every op
// in the projection family (linear, linear_add, the fused attention input split, and the fused GDN
// split) is this kernel with a different epilogue.
constexpr int kMpBlockRows = 128;
constexpr int kMpBlockCols = 128;
constexpr int kMpBlockK = 64;
constexpr int kMpWarpRows = 64;
constexpr int kMpWarpCols = 32;
constexpr int kMpMi = kMpWarpRows / 16;
constexpr int kMpNi = kMpWarpCols / 8;
constexpr int kMpThreads = 256;
constexpr int kMpWarps = kMpThreads / 32;
constexpr int kMpActivationItems = kMpBlockCols * (kMpBlockK / 8) / kMpThreads;
constexpr int kMpWeightItems = kMpBlockRows * (kMpBlockK / 32) * 2 / kMpThreads;
constexpr int kMpSharedBytes =
    (kMpBlockRows + kMpBlockCols) * kMpBlockK * static_cast<int>(sizeof(__half));

static_assert(kMpWarps == (kMpBlockRows / kMpWarpRows) * (kMpBlockCols / kMpWarpCols));
static_assert(kMpWeightItems * kMpThreads == kMpBlockRows * (kMpBlockK / 32) * 2);
static_assert(kMpActivationItems * kMpThreads == kMpBlockCols * (kMpBlockK / 8));
static_assert(kMpSharedBytes + 1024 <= 64 * 1024);

// Per-tile state and body, shared between the flat-grid and persistent kernels below.
// The flat-grid kernel calls it once per CTA; the persistent kernel calls it in a stride
// loop over all (row_block, col_block) tile pairs assigned to this CTA.
struct MpTileState {
    int row0;
    int col0;
    int tokens;
    int output_rows;
    int k_tiles;
    int k_groups;
    int n_tiles;
    const __nv_bfloat16* activation_sources[kMpActivationItems];
    int activation_destinations[kMpActivationItems];
    const std::uint8_t* weight_sources[kMpWeightItems];
    std::int64_t weight_stride;
};

__device__ __forceinline__ void mp_init_tile_state(
    MpTileState& s, int row0, int col0, int output_rows, int input_rows, int tokens,
    const __nv_bfloat16* x, const std::uint8_t* codes, int tid) {
    s.row0 = row0;
    s.col0 = col0;
    s.tokens = tokens;
    s.output_rows = output_rows;
    s.k_tiles = input_rows / kMpBlockK;
    s.k_groups = input_rows / 128;
    s.n_tiles = output_rows / 32;
#pragma unroll
    for (int i = 0; i < kMpWeightItems; ++i) {
        const int unit_slot = tid + i * kMpThreads;
        const int unit_tile = unit_slot >> 6;
        const int unit_in_tile = unit_slot & 63;
        const int unit_k_half = unit_tile >> 2;
        const int unit_n_tile = unit_tile & 3;
        const int unit_t = unit_in_tile >> 1;
        const int unit_w = (unit_in_tile & 1) * 2;
        s.weight_sources[i] =
            codes + static_cast<std::int64_t>(unit_k_half * s.n_tiles + row0 / 32 +
                                              unit_n_tile) * 1024 +
            32 * unit_t + 8 * unit_w;
    }
    s.weight_stride = static_cast<std::int64_t>(2) * s.n_tiles * 1024;
}

template <class Output>
__device__ __forceinline__ void mp_run_tile_body(
    MpTileState& s, const unsigned* fp8_code_table, __half* shared, int tid, int lane,
    int warp_row, int warp_col, int a_row_offset, int a_col_offset, int b_inner_row,
    int b_k_offset, const int (&weight_destinations)[kMpWeightItems][4],
    const __nv_bfloat16* scales, Output& output) {
    __half* weights = shared;
    __half* activations = shared + kMpBlockRows * kMpBlockK;

    float accum[kMpMi][kMpNi][4] = {};
    unsigned partial[kMpMi][kMpNi][2] = {};

    // The staging loads for the next K tile are issued before the current tile's mma and
    // staged after it, so one tile's global latency and its copy/decode instructions run
    // under the other's tensor work.
    uint4 unit[kMpWeightItems];
    int4 packed[kMpActivationItems];
    __half2 half_scale[kMpWeightItems];
#pragma unroll
    for (int i = 0; i < kMpWeightItems; ++i) {
        half_scale[i] = __half2half2(__float2half_rn(__bfloat162float(
            scales[fp8_block_scale_index(s.row0, 0, s.k_groups)])));
        unit[i] = load_ldg<uint4>(reinterpret_cast<const uint4*>(s.weight_sources[i]));
    }
#pragma unroll
    for (int i = 0; i < kMpActivationItems; ++i) {
        const int token = s.col0 + (tid + i * kMpThreads) / (kMpBlockK / 8);
        packed[i] = token < s.tokens
                    ? load_ldg<int4>(s.activation_sources[i])
                    : make_int4(0, 0, 0, 0);
    }

    for (int tile = 0; tile < s.k_tiles; ++tile) {
        const int next = tile + 1;
        uint4 next_unit[kMpWeightItems];
        int4 next_packed[kMpActivationItems];
        __half2 next_scale[kMpWeightItems];
        if (next < s.k_tiles) {
#pragma unroll
            for (int i = 0; i < kMpWeightItems; ++i) {
                next_scale[i] = __half2half2(__float2half_rn(__bfloat162float(
                    scales[fp8_block_scale_index(s.row0, next >> 1, s.k_groups)])));
                next_unit[i] = load_ldg<uint4>(reinterpret_cast<const uint4*>(
                    s.weight_sources[i] + next * s.weight_stride));
            }
#pragma unroll
            for (int i = 0; i < kMpActivationItems; ++i) {
                const int token = s.col0 + (tid + i * kMpThreads) / (kMpBlockK / 8);
                next_packed[i] = token < s.tokens
                                 ? load_ldg<int4>(s.activation_sources[i] + next * kMpBlockK)
                                 : make_int4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int i = 0; i < kMpWeightItems; ++i) {
            store_vec(weights + weight_destinations[i][0],
                      decode_marlin_quad(fp8_code_table, unit[i].x, half_scale[i]));
            store_vec(weights + weight_destinations[i][1],
                      decode_marlin_quad(fp8_code_table, unit[i].y, half_scale[i]));
            store_vec(weights + weight_destinations[i][2],
                      decode_marlin_quad(fp8_code_table, unit[i].z, half_scale[i]));
            store_vec(weights + weight_destinations[i][3],
                      decode_marlin_quad(fp8_code_table, unit[i].w, half_scale[i]));
        }
#pragma unroll
        for (int i = 0; i < kMpActivationItems; ++i) {
            stage_bf16_word(activations + s.activation_destinations[i], packed[i]);
        }
        __syncthreads();

#pragma unroll
        for (int k_sub = 0; k_sub < kMpBlockK / 16; ++k_sub) {
            const int k_step = k_sub * 16;
            unsigned b_fragments[kMpNi][2];
#pragma unroll
            for (int ni = 0; ni < kMpNi; ++ni) {
                const int row = warp_col * kMpWarpCols + ni * 8 + b_inner_row;
                const int column = k_step + b_k_offset;
                ldmatrix_x2(b_fragments[ni][0], b_fragments[ni][1],
                            smem_addr(activations + row * kMpBlockK +
                                      fp8_block_swizzle(row, column)));
            }
#pragma unroll
            for (int mi = 0; mi < kMpMi; ++mi) {
                unsigned a_fragments[4];
                const int row = warp_row * kMpWarpRows + mi * 16 + a_row_offset;
                const int column = k_step + a_col_offset;
                ldmatrix_x4(a_fragments[0], a_fragments[1], a_fragments[2], a_fragments[3],
                            smem_addr(weights + row * kMpBlockK +
                                      fp8_block_swizzle(row, column)));
#pragma unroll
                for (int ni = 0; ni < kMpNi; ++ni) {
                    fp8_mma_f16_f16acc(partial[mi][ni], a_fragments[0], a_fragments[1],
                                       a_fragments[2], a_fragments[3], b_fragments[ni][0],
                                       b_fragments[ni][1]);
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < kMpMi; ++mi) {
#pragma unroll
            for (int ni = 0; ni < kMpNi; ++ni) {
                fp8_fold_f16_partial(accum[mi][ni], partial[mi][ni]);
            }
        }
        __syncthreads();
        if (next < s.k_tiles) {
#pragma unroll
            for (int i = 0; i < kMpWeightItems; ++i) {
                unit[i] = next_unit[i];
                half_scale[i] = next_scale[i];
            }
#pragma unroll
            for (int i = 0; i < kMpActivationItems; ++i) { packed[i] = next_packed[i]; }
        }
    }

    const int mma_row = lane >> 2;
    const int mma_col = 2 * (lane & 3);
#pragma unroll
    for (int mi = 0; mi < kMpMi; ++mi) {
        const int row_a = s.row0 + warp_row * kMpWarpRows + mi * 16 + mma_row;
        const int row_b = row_a + 8;
#pragma unroll
        for (int ni = 0; ni < kMpNi; ++ni) {
            const int col_a = s.col0 + warp_col * kMpWarpCols + ni * 8 + mma_col;
            const int col_b = col_a + 1;
            const float* values = accum[mi][ni];
            if (row_a < s.output_rows) {
                if (col_a < s.tokens) { output.store(row_a, col_a, values[0]); }
                if (col_b < s.tokens) { output.store(row_a, col_b, values[1]); }
            }
            if (row_b < s.output_rows) {
                if (col_a < s.tokens) { output.store(row_b, col_a, values[2]); }
                if (col_b < s.tokens) { output.store(row_b, col_b, values[3]); }
            }
        }
    }
}

template <class Output>
__global__ __launch_bounds__(kMpThreads, 1) void marlin_fp8_block_linear_hmma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, Output output, int output_rows, int input_rows,
    int tokens) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp >> 2;
    const int warp_col = warp & 3;
    const int row0 = static_cast<int>(blockIdx.x) * kMpBlockRows;
    const int col0 = static_cast<int>(blockIdx.y) * kMpBlockCols;

    const int a_matrix = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;

    // One aligned 16-code unit per thread per K tile: the tile's 512 units are handed out
    // thread-major, so a warp copies 512 contiguous bytes of one 32x32 code tile.
    int weight_destinations[kMpWeightItems][4];
#pragma unroll
    for (int i = 0; i < kMpWeightItems; ++i) {
        const int unit_slot = tid + i * kMpThreads;
        const int unit_tile = unit_slot >> 6;
        const int unit_in_tile = unit_slot & 63;
        const int unit_k_half = unit_tile >> 2;
        const int unit_n_tile = unit_tile & 3;
        const int unit_t = unit_in_tile >> 1;
        const int unit_w = (unit_in_tile & 1) * 2;
        const int unit_row = unit_n_tile * 32 + (unit_w >> 1) * 16 + (unit_t >> 2);
        const int unit_column = unit_k_half * 32 + (unit_t & 3) * 4;
        weight_destinations[i][0] = unit_row * kMpBlockK + fp8_block_swizzle(unit_row, unit_column);
        weight_destinations[i][1] =
            unit_row * kMpBlockK + fp8_block_swizzle(unit_row, unit_column + 16);
        weight_destinations[i][2] =
            (unit_row + 8) * kMpBlockK + fp8_block_swizzle(unit_row + 8, unit_column);
        weight_destinations[i][3] =
            (unit_row + 8) * kMpBlockK + fp8_block_swizzle(unit_row + 8, unit_column + 16);
    }

    MpTileState state;
    mp_init_tile_state(state, row0, col0, output_rows, input_rows, tokens, x, codes, tid);
#pragma unroll
    for (int i = 0; i < kMpActivationItems; ++i) {
        const int item = tid + i * kMpThreads;
        const int row = item / (kMpBlockK / 8);
        const int k8 = item - row * (kMpBlockK / 8);
        state.activation_destinations[i] = row * kMpBlockK + fp8_block_swizzle(row, k8 * 8);
        state.activation_sources[i] =
            x + static_cast<std::int64_t>(col0 + row) * input_rows + k8 * 8;
    }

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    mp_run_tile_body(state, fp8_code_table, shared, tid, lane, warp_row, warp_col, a_row_offset,
                     a_col_offset, b_inner_row, b_k_offset, weight_destinations, scales, output);
}

// Persistent-CTA sibling: 1D grid of `gridDim.x` CTAs that stride through all (row*col) tile
// pairs. gridDim.x is clamped to the device's SM count by the launcher, so each SM keeps one
// resident CTA busy for the whole kernel. The per-tile body is identical to the flat-grid
// kernel above - the only difference is the outer loop.
template <class Output>
__global__ __launch_bounds__(kMpThreads, 1) void marlin_fp8_block_linear_hmma_kernel_persistent(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, Output output, int output_rows, int input_rows,
    int tokens, int total_tiles) {
    extern __shared__ __align__(16) __half shared[];
    __shared__ unsigned fp8_code_table[kFp8CodeCount];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_row = warp >> 2;
    const int warp_col = warp & 3;

    const int a_matrix = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;

    int weight_destinations[kMpWeightItems][4];
#pragma unroll
    for (int i = 0; i < kMpWeightItems; ++i) {
        const int unit_slot = tid + i * kMpThreads;
        const int unit_tile = unit_slot >> 6;
        const int unit_in_tile = unit_slot & 63;
        const int unit_k_half = unit_tile >> 2;
        const int unit_n_tile = unit_tile & 3;
        const int unit_t = unit_in_tile >> 1;
        const int unit_w = (unit_in_tile & 1) * 2;
        const int unit_row = unit_n_tile * 32 + (unit_w >> 1) * 16 + (unit_t >> 2);
        const int unit_column = unit_k_half * 32 + (unit_t & 3) * 4;
        weight_destinations[i][0] = unit_row * kMpBlockK + fp8_block_swizzle(unit_row, unit_column);
        weight_destinations[i][1] =
            unit_row * kMpBlockK + fp8_block_swizzle(unit_row, unit_column + 16);
        weight_destinations[i][2] =
            (unit_row + 8) * kMpBlockK + fp8_block_swizzle(unit_row + 8, unit_column);
        weight_destinations[i][3] =
            (unit_row + 8) * kMpBlockK + fp8_block_swizzle(unit_row + 8, unit_column + 16);
    }

    const int row_blocks = output_rows / kMpBlockRows;
    const int col_blocks = (tokens + kMpBlockCols - 1) / kMpBlockCols;

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    MpTileState state;
    for (int linear_id = blockIdx.x; linear_id < total_tiles; linear_id += gridDim.x) {
        const int row_block = linear_id / col_blocks;
        const int col_block = linear_id - row_block * col_blocks;
        const int row0 = row_block * kMpBlockRows;
        const int col0 = col_block * kMpBlockCols;

        mp_init_tile_state(state, row0, col0, output_rows, input_rows, tokens, x, codes, tid);
#pragma unroll
        for (int i = 0; i < kMpActivationItems; ++i) {
            const int item = tid + i * kMpThreads;
            const int row = item / (kMpBlockK / 8);
            const int k8 = item - row * (kMpBlockK / 8);
            state.activation_destinations[i] = row * kMpBlockK + fp8_block_swizzle(row, k8 * 8);
            state.activation_sources[i] =
                x + static_cast<std::int64_t>(col0 + row) * input_rows + k8 * 8;
        }
        mp_run_tile_body(state, fp8_code_table, shared, tid, lane, warp_row, warp_col, a_row_offset,
                         a_col_offset, b_inner_row, b_k_offset, weight_destinations, scales,
                         output);
        __syncthreads();
    }
}

template <class Output>
void launch_marlin_projection_persistent(const Tensor& x, const Weight& weight, Output output,
                                         cudaStream_t stream) {
    const int row_blocks = weight.n / kMpBlockRows;
    const int col_blocks = (x.ne[1] + kMpBlockCols - 1) / kMpBlockCols;
    const int total_tiles = row_blocks * col_blocks;
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    const int grid = std::min(total_tiles, sm_count);
    ensure_func_attr_per_device(marlin_fp8_block_linear_hmma_kernel_persistent<Output>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, kMpSharedBytes);
    marlin_fp8_block_linear_hmma_kernel_persistent<Output>
        <<<grid, kMpThreads, kMpSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, weight.n, weight.k, x.ne[1],
            total_tiles);
    CUDA_CHECK(cudaGetLastError());
}

template <class Output>
void launch_marlin_projection(const Tensor& x, const Weight& weight, Output output,
                              cudaStream_t stream) {
    if (std::getenv("NINFER_FP8_SHAPE_LOG") != nullptr) {
        std::fprintf(stderr, "proj n=%d k=%d tokens=%d\n", weight.n, weight.k, x.ne[1]);
    }
    const dim3 grid(static_cast<unsigned>(weight.n / kMpBlockRows),
                    static_cast<unsigned>((x.ne[1] + kMpBlockCols - 1) / kMpBlockCols));
    ensure_func_attr_per_device(marlin_fp8_block_linear_hmma_kernel<Output>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, kMpSharedBytes);
    marlin_fp8_block_linear_hmma_kernel<Output>
        <<<grid, kMpThreads, kMpSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, weight.n, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// --- Marlin persistent-layout small-token leaf -------------------------------------------------
//
// Below the tile crossover the tile kernel is the wrong organization: a 128-token weight tile is
// filled with one to eight tokens, so almost every mma column and every shared-memory round trip is
// wasted, and the 33 KiB weight tile caps residency at one CTA per SM. The decode leaf keeps the
// same coalesced weight path - one aligned 16-byte unit per thread per K tile, thread-major over
// the tile's 64 units - but stops there: it decodes the unit's four 4-code groups into registers,
// multiplies them with the token activations, and never stages a weight byte, so the kernel needs
// only the 1 KiB code table in shared memory. One unit covers two rows (unit_row, unit_row + 8) and
// the four k groups of its 32-wide K half, so a row's partial sums live in the four lanes that
// differ in the low two k-group bits; two warp shuffles after the K loop combine them.
constexpr int kMdBlockRows = 128;
constexpr int kMdBlockK = 64;
constexpr int kMdThreads = 256;
constexpr int kMdMaxTokens = 16;
constexpr int kMdSwiGluMaxTokens = 8;

static_assert(kMdThreads == (kMdBlockRows / 32) * 64);

// One 16-byte unit holds the codes of two rows (unit_row, unit_row + 8): its first 8 bytes are the
// low 16-wide K half of one 32-wide half and the second 8 bytes the same K columns eight rows
// further down, so a unit only ever spans half of the 64-wide K tile a thread consumes. Each thread
// therefore takes the same unit slot in both K halves, which puts all of a row's K coverage in the
// four lanes that differ in the low two k-group bits; two warp shuffles combine them.
template <int Tokens>
__device__ __forceinline__ void marlin_decode_accumulate(
    const unsigned* table, uint4 raw, __half2 half_scale, const __nv_bfloat16* low_activations,
    const __nv_bfloat16* high_activations, std::int64_t token_stride, int tokens,
    float (&accum)[Tokens][2]) {
    const uint2 first_low = decode_marlin_quad(table, raw.x, half_scale);
    const uint2 first_high = decode_marlin_quad(table, raw.y, half_scale);
    const uint2 second_low = decode_marlin_quad(table, raw.z, half_scale);
    const uint2 second_high = decode_marlin_quad(table, raw.w, half_scale);
    const float2 code0 = __half22float2(*reinterpret_cast<const __half2*>(&first_low.x));
    const float2 code1 = __half22float2(*reinterpret_cast<const __half2*>(&first_low.y));
    const float2 code2 = __half22float2(*reinterpret_cast<const __half2*>(&first_high.x));
    const float2 code3 = __half22float2(*reinterpret_cast<const __half2*>(&first_high.y));
    const float2 paired0 = __half22float2(*reinterpret_cast<const __half2*>(&second_low.x));
    const float2 paired1 = __half22float2(*reinterpret_cast<const __half2*>(&second_low.y));
    const float2 paired2 = __half22float2(*reinterpret_cast<const __half2*>(&second_high.x));
    const float2 paired3 = __half22float2(*reinterpret_cast<const __half2*>(&second_high.y));
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const std::int64_t offset = static_cast<std::int64_t>(token) * token_stride;
        const uint2 packed_low =
            token < tokens
                ? load_ldg<uint2>(reinterpret_cast<const uint2*>(low_activations + offset))
                : make_uint2(0U, 0U);
        const uint2 packed_high =
            token < tokens
                ? load_ldg<uint2>(reinterpret_cast<const uint2*>(high_activations + offset))
                : make_uint2(0U, 0U);
        const float2 activation0 = bf16x2_wide(packed_low.x);
        const float2 activation1 = bf16x2_wide(packed_low.y);
        const float2 activation2 = bf16x2_wide(packed_high.x);
        const float2 activation3 = bf16x2_wide(packed_high.y);
        accum[token][0] += code0.x * activation0.x + code0.y * activation0.y +
                           code1.x * activation1.x + code1.y * activation1.y +
                           code2.x * activation2.x + code2.y * activation2.y +
                           code3.x * activation3.x + code3.y * activation3.y;
        accum[token][1] += paired0.x * activation0.x + paired0.y * activation0.y +
                           paired1.x * activation1.x + paired1.y * activation1.y +
                           paired2.x * activation2.x + paired2.y * activation2.y +
                           paired3.x * activation3.x + paired3.y * activation3.y;
    }
}

template <class Output, int Tokens>
__global__ __launch_bounds__(kMdThreads) void marlin_fp8_block_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, Output output, int output_rows, int input_rows,
    int tokens) {
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    const int tid = static_cast<int>(threadIdx.x);
    const int row0 = static_cast<int>(blockIdx.x) * kMdBlockRows;
    const int k_tiles = input_rows / kMdBlockK;
    const int n_tiles = output_rows / 32;
    const int k_groups = input_rows / 128;

    const int unit_n_tile = tid >> 6;
    const int unit_in_tile = tid & 63;
    const int unit_t = unit_in_tile >> 1;
    const int unit_w = (unit_in_tile & 1) * 2;
    const int unit_row = unit_n_tile * 32 + (unit_w >> 1) * 16 + (unit_t >> 2);
    const int unit_column = (unit_t & 3) * 4;
    const std::uint8_t* weight_source =
        codes + static_cast<std::int64_t>(row0 / 32 + unit_n_tile) * 1024 + 32 * unit_t +
        8 * unit_w;
    const std::int64_t weight_source_stride = static_cast<std::int64_t>(2) * n_tiles * 1024;
    const std::int64_t half_stride = static_cast<std::int64_t>(n_tiles) * 1024;
    const __nv_bfloat16* activation_low = x + unit_column;
    const __nv_bfloat16* activation_high = x + unit_column + 16;

    float accum[Tokens][2] = {};

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    for (int tile = 0; tile < k_tiles; ++tile) {
        const std::int64_t tile_offset = static_cast<std::int64_t>(tile) * kMdBlockK;
        const __half2 half_scale = __half2half2(__float2half_rn(__bfloat162float(
            scales[fp8_block_scale_index(row0, tile >> 1, k_groups)])));
        const std::uint8_t* source = weight_source + tile * weight_source_stride;
        marlin_decode_accumulate<Tokens>(
            fp8_code_table, load_ldg<uint4>(reinterpret_cast<const uint4*>(source)), half_scale,
            activation_low + tile_offset, activation_high + tile_offset, input_rows, tokens, accum);
        marlin_decode_accumulate<Tokens>(
            fp8_code_table, load_ldg<uint4>(reinterpret_cast<const uint4*>(source + half_stride)),
            half_scale, activation_low + tile_offset + 32, activation_high + tile_offset + 32,
            input_rows, tokens, accum);
    }

#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
#pragma unroll
        for (int offset = 2; offset <= 4; offset <<= 1) {
            accum[token][0] += __shfl_xor_sync(0xffffffffU, accum[token][0], offset);
            accum[token][1] += __shfl_xor_sync(0xffffffffU, accum[token][1], offset);
        }
    }

    if ((unit_t & 3) == 0) {
        const int row_low = row0 + unit_row;
        const int row_high = row_low + 8;
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (token < tokens) {
                output.store(row_low, token, accum[token][0]);
                output.store(row_high, token, accum[token][1]);
            }
        }
    }
}

// The fused SwiGLU weight interleaves a gate row r with its up row r + intermediate_rows, and the
// up band is tile aligned in the persistent layout (intermediate_rows is a multiple of 32), so the
// decode leaf reads the matching up unit of every gate unit and combines the two accumulated dot
// products in the epilogue.
template <int Tokens>
__global__ __launch_bounds__(kMdThreads) void marlin_fp8_block_swiglu_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int intermediate_rows, int total_rows, int input_rows, int tokens) {
    __shared__ unsigned fp8_code_table[kFp8CodeCount];
    const int tid = static_cast<int>(threadIdx.x);
    const int row0 = static_cast<int>(blockIdx.x) * kMdBlockRows;
    const int k_tiles = input_rows / kMdBlockK;
    const int n_tiles = total_rows / 32;
    const int k_groups = input_rows / 128;

    const int unit_n_tile = tid >> 6;
    const int unit_in_tile = tid & 63;
    const int unit_t = unit_in_tile >> 1;
    const int unit_w = (unit_in_tile & 1) * 2;
    const int unit_row = unit_n_tile * 32 + (unit_w >> 1) * 16 + (unit_t >> 2);
    const int unit_column = (unit_t & 3) * 4;
    const std::uint8_t* gate_source =
        codes + static_cast<std::int64_t>(row0 / 32 + unit_n_tile) * 1024 + 32 * unit_t +
        8 * unit_w;
    const std::int64_t weight_source_stride = static_cast<std::int64_t>(2) * n_tiles * 1024;
    const std::int64_t half_stride = static_cast<std::int64_t>(n_tiles) * 1024;
    const std::int64_t up_offset = static_cast<std::int64_t>(intermediate_rows / 32) * 1024;
    const __nv_bfloat16* activation_low = x + unit_column;
    const __nv_bfloat16* activation_high = x + unit_column + 16;

    float gate_accum[Tokens][2] = {};
    float up_accum[Tokens][2] = {};

    build_fp8_code_table(fp8_code_table, tid);
    __syncthreads();

    for (int tile = 0; tile < k_tiles; ++tile) {
        const std::int64_t tile_offset = static_cast<std::int64_t>(tile) * kMdBlockK;
        const int k_group = tile >> 1;
        const __half2 gate_scale = __half2half2(__float2half_rn(
            __bfloat162float(scales[fp8_block_scale_index(row0, k_group, k_groups)])));
        const __half2 up_scale = __half2half2(__float2half_rn(__bfloat162float(
            scales[fp8_block_scale_index(intermediate_rows + row0, k_group, k_groups)])));
        const std::uint8_t* source = gate_source + tile * weight_source_stride;
        marlin_decode_accumulate<Tokens>(
            fp8_code_table, load_ldg<uint4>(reinterpret_cast<const uint4*>(source)), gate_scale,
            activation_low + tile_offset, activation_high + tile_offset, input_rows, tokens,
            gate_accum);
        marlin_decode_accumulate<Tokens>(
            fp8_code_table, load_ldg<uint4>(reinterpret_cast<const uint4*>(source + up_offset)),
            up_scale, activation_low + tile_offset, activation_high + tile_offset, input_rows,
            tokens, up_accum);
        marlin_decode_accumulate<Tokens>(
            fp8_code_table, load_ldg<uint4>(reinterpret_cast<const uint4*>(source + half_stride)),
            gate_scale, activation_low + tile_offset + 32, activation_high + tile_offset + 32,
            input_rows, tokens, gate_accum);
        marlin_decode_accumulate<Tokens>(
            fp8_code_table,
            load_ldg<uint4>(reinterpret_cast<const uint4*>(source + half_stride + up_offset)),
            up_scale, activation_low + tile_offset + 32, activation_high + tile_offset + 32,
            input_rows, tokens, up_accum);
    }

#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
#pragma unroll
        for (int offset = 2; offset <= 4; offset <<= 1) {
            gate_accum[token][0] += __shfl_xor_sync(0xffffffffU, gate_accum[token][0], offset);
            gate_accum[token][1] += __shfl_xor_sync(0xffffffffU, gate_accum[token][1], offset);
            up_accum[token][0] += __shfl_xor_sync(0xffffffffU, up_accum[token][0], offset);
            up_accum[token][1] += __shfl_xor_sync(0xffffffffU, up_accum[token][1], offset);
        }
    }

    if ((unit_t & 3) == 0) {
        const int row_low = row0 + unit_row;
        const int row_high = row_low + 8;
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (token < tokens) {
                out[static_cast<std::int64_t>(token) * intermediate_rows + row_low] =
                    __float2bfloat16_rn(silu(gate_accum[token][0]) * up_accum[token][0]);
                out[static_cast<std::int64_t>(token) * intermediate_rows + row_high] =
                    __float2bfloat16_rn(silu(gate_accum[token][1]) * up_accum[token][1]);
            }
        }
    }
}

template <int Tokens>
void launch_swiglu_decode_case(dim3 grid, cudaStream_t stream, const __nv_bfloat16* activation,
                               const std::uint8_t* codes, const __nv_bfloat16* scales,
                               __nv_bfloat16* output, int intermediate_rows, int total_rows,
                               int input_rows, int tokens) {
    marlin_fp8_block_swiglu_decode_kernel<Tokens><<<grid, kMdThreads, 0, stream>>>(
        activation, codes, scales, output, intermediate_rows, total_rows, input_rows, tokens);
}

void launch_swiglu_marlin_decode(const Tensor& x, const Weight& weight, Tensor& out,
                                 int intermediate_rows, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(intermediate_rows / kMdBlockRows));
    const int tokens = x.ne[1];
    const __nv_bfloat16* activation = static_cast<const __nv_bfloat16*>(x.data);
    const std::uint8_t* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const __nv_bfloat16* scales = static_cast<const __nv_bfloat16*>(weight.scales);
    auto* output = static_cast<__nv_bfloat16*>(out.data);
    if (tokens <= 1) {
        launch_swiglu_decode_case<1>(grid, stream, activation, codes, scales, output,
                                     intermediate_rows, weight.n, weight.k, tokens);
    } else if (tokens <= 2) {
        launch_swiglu_decode_case<2>(grid, stream, activation, codes, scales, output,
                                     intermediate_rows, weight.n, weight.k, tokens);
    } else if (tokens <= 4) {
        launch_swiglu_decode_case<4>(grid, stream, activation, codes, scales, output,
                                     intermediate_rows, weight.n, weight.k, tokens);
    } else {
        launch_swiglu_decode_case<8>(grid, stream, activation, codes, scales, output,
                                     intermediate_rows, weight.n, weight.k, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Output>
void launch_marlin_decode(const Tensor& x, const Weight& weight, Output output,
                          cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(weight.n / kMdBlockRows));
    const int tokens = x.ne[1];
    const __nv_bfloat16* activation = static_cast<const __nv_bfloat16*>(x.data);
    const std::uint8_t* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const __nv_bfloat16* scales = static_cast<const __nv_bfloat16*>(weight.scales);
#define NINFER_MARLIN_DECODE_CASE(Tokens)                                                        \
    marlin_fp8_block_decode_kernel<Output, Tokens>                                               \
        <<<grid, kMdThreads, 0, stream>>>(activation, codes, scales, output, weight.n, weight.k, \
                                          tokens)
    if (tokens <= 1) {
        NINFER_MARLIN_DECODE_CASE(1);
    } else if (tokens <= 2) {
        NINFER_MARLIN_DECODE_CASE(2);
    } else if (tokens <= 4) {
        NINFER_MARLIN_DECODE_CASE(4);
    } else if (tokens <= 8) {
        NINFER_MARLIN_DECODE_CASE(8);
    } else {
        NINFER_MARLIN_DECODE_CASE(16);
    }
#undef NINFER_MARLIN_DECODE_CASE
    CUDA_CHECK(cudaGetLastError());
}

template <class Output>
bool launch_marlin_leaf(const Tensor& x, const Weight& weight, Output output,
                        cudaStream_t stream) {
    // The Marlin lane has exactly two leaves: the small-token decode leaf below kMdMaxTokens, and
    // the 128-token tile kernel above it. The row-major scalar/HMMA crossover does not apply here -
    // the tile kernel is the Marlin lane's high-token leaf, and the scalar reader stays available
    // for the sub-128-row shapes and as a diagnostic override.
    if ((weight.n % kMpBlockRows) != 0) { return false; }
    const Fp8BlockRoute route = fp8_block_route();
    if (route == Fp8BlockRoute::Scalar) { return false; }
    if (x.ne[1] <= kMdMaxTokens && route != Fp8BlockRoute::Hmma) {
        launch_marlin_decode(x, weight, output, stream);
        return true;
    }
    if (fp8_block_persistent()) {
        launch_marlin_projection_persistent(x, weight, output, stream);
    } else {
        launch_marlin_projection(x, weight, output, stream);
    }
    return true;
}

bool validate_projection_input(const Tensor& x, const Weight& weight, const char* operation) {
    const bool marlin = weight.layout == QuantLayout::MarlinFp8Block128;
    if (marlin) {
        (void)validate_marlin_fp8_block_weight(weight, operation);
    } else {
        (void)validate_fp8_block_weight(weight, operation);
    }
    if (x.dtype != DType::BF16 || x.ne[0] != weight.k || x.ne[1] <= 0 || x.ne[2] != 1 ||
        x.ne[3] != 1 || !x.is_contiguous() || x.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(x.data) & 15U) != 0) {
        throw std::invalid_argument(std::string(operation) + ": invalid input tensor");
    }
    return marlin;
}

} // namespace

void fp8_block_linear_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    const bool marlin = validate_projection_input(x, weight, "block-FP8 linear");
    if (out.dtype != DType::BF16 || out.ne[0] != weight.n || out.ne[1] != x.ne[1] ||
        out.ne[2] != 1 || out.ne[3] != 1 || !out.is_contiguous() || out.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(out.data) & 15U) != 0) {
        throw std::invalid_argument("block-FP8 linear: invalid output tensor");
    }
    if (marlin) {
        if (!launch_marlin_leaf(
                x, weight, LinearOutput{static_cast<__nv_bfloat16*>(out.data), weight.n},
                stream)) {
            launch<LinearOutput, true>(
                x, weight, LinearOutput{static_cast<__nv_bfloat16*>(out.data), weight.n}, stream);
        }
        return;
    }
    if (use_hmma(x, weight, Fp8BlockOperation::Linear)) {
        launch_hmma(x, weight,
                    LinearOutput{static_cast<__nv_bfloat16*>(out.data), weight.n}, stream);
    } else {
        launch(x, weight, LinearOutput{static_cast<__nv_bfloat16*>(out.data), weight.n}, stream);
    }
}

void fp8_block_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    const bool marlin = validate_projection_input(x, weight, "block-FP8 linear_add");
    if (residual.dtype != DType::BF16 || residual.ne[0] != weight.n ||
        residual.ne[1] != x.ne[1] || residual.ne[2] != 1 || residual.ne[3] != 1 ||
        !residual.is_contiguous() || residual.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(residual.data) & 15U) != 0) {
        throw std::invalid_argument("block-FP8 linear_add: invalid residual tensor");
    }
    const LinearAddOutput output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    if (marlin) {
        if (!launch_marlin_leaf(x, weight, output, stream)) {
            launch<LinearAddOutput, true>(x, weight, output, stream);
        }
        return;
    }
    if (use_hmma(x, weight, Fp8BlockOperation::LinearAdd)) {
        launch_hmma(x, weight, output, stream);
    } else {
        launch(x, weight, output, stream);
    }
}

template <bool Marlin, int TokensPerBlock>
void launch_swiglu_scalar_tokens(const Tensor& x, const Weight& weight, Tensor& out,
                                 int intermediate_rows, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(intermediate_rows / kRowsPerBlock),
                    static_cast<unsigned>((x.ne[1] + TokensPerBlock - 1) / TokensPerBlock));
    fp8_block_linear_swiglu_kernel<Marlin, TokensPerBlock><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        intermediate_rows, weight.k, x.ne[1]);
}

void fp8_block_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    const bool marlin = validate_projection_input(x, weight, "block-FP8 linear_swiglu");
    if ((weight.n != 34816 && weight.n != 17408) || weight.k != 5120 ||
        (weight.n % (2 * kBlockRows)) != 0) {
        throw std::invalid_argument("block-FP8 linear_swiglu: unsupported weight shape");
    }
    const int intermediate_rows = weight.n / 2;
    if (out.dtype != DType::BF16 || out.ne[0] != intermediate_rows || out.ne[1] != x.ne[1] ||
        out.ne[2] != 1 || out.ne[3] != 1 || !out.is_contiguous() || out.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(out.data) & 15U) != 0) {
        throw std::invalid_argument("block-FP8 linear_swiglu: invalid output tensor");
    }
    if (marlin) {
        const Fp8BlockRoute route = fp8_block_route();
        if (route != Fp8BlockRoute::Scalar && (intermediate_rows % kMdBlockRows) == 0) {
            if (x.ne[1] <= kMdSwiGluMaxTokens && route != Fp8BlockRoute::Hmma) {
                launch_swiglu_marlin_decode(x, weight, out, intermediate_rows, stream);
                return;
            }
            launch_swiglu_marlin(x, weight, out, intermediate_rows, stream);
            return;
        }
        // Diagnostic scalar override, or an intermediate row count the unit leaves cannot tile.
        if (x.ne[1] <= 4) {
            launch_swiglu_scalar_tokens<true, 4>(x, weight, out, intermediate_rows, stream);
        } else if (x.ne[1] <= 5) {
            launch_swiglu_scalar_tokens<true, 5>(x, weight, out, intermediate_rows, stream);
        } else if (x.ne[1] <= 6) {
            launch_swiglu_scalar_tokens<true, 6>(x, weight, out, intermediate_rows, stream);
        } else if (x.ne[1] <= 7) {
            launch_swiglu_scalar_tokens<true, 7>(x, weight, out, intermediate_rows, stream);
        } else {
            launch_swiglu_scalar_tokens<true, 8>(x, weight, out, intermediate_rows, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (use_hmma(x, weight, Fp8BlockOperation::SwiGlu)) {
        launch_swiglu_hmma(x, weight, out, intermediate_rows, stream);
        return;
    }
    if (x.ne[1] <= 4) {
        launch_swiglu_scalar_tokens<false, 4>(x, weight, out, intermediate_rows, stream);
    } else if (x.ne[1] <= 5) {
        launch_swiglu_scalar_tokens<false, 5>(x, weight, out, intermediate_rows, stream);
    } else if (x.ne[1] <= 6) {
        launch_swiglu_scalar_tokens<false, 6>(x, weight, out, intermediate_rows, stream);
    } else if (x.ne[1] <= 7) {
        launch_swiglu_scalar_tokens<false, 7>(x, weight, out, intermediate_rows, stream);
    } else {
        launch_swiglu_scalar_tokens<false, 8>(x, weight, out, intermediate_rows, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void fp8_block_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& query,
                                  Tensor& gate, Tensor& key, Tensor& value,
                                  cudaStream_t stream) {
    const bool marlin =
        validate_projection_input(x, weight, "block-FP8 attention projection");
    if ((weight.n != 14336 && weight.n != 7168) || weight.k != 5120) {
        throw std::invalid_argument("block-FP8 attention projection: unsupported weight shape");
    }
    const int query_rows = weight.n == 14336 ? 6144 : 3072;
    const int key_value_rows = weight.n == 14336 ? 1024 : 512;
    const auto valid_output = [&](const Tensor& output, int rows) {
        return output.dtype == DType::BF16 && output.ne[0] == rows && output.ne[1] == x.ne[1] &&
               output.ne[2] == 1 && output.ne[3] == 1 && output.is_contiguous() &&
               output.data != nullptr && (reinterpret_cast<std::uintptr_t>(output.data) & 15U) == 0;
    };
    if (!valid_output(query, query_rows) || !valid_output(gate, query_rows) ||
        !valid_output(key, key_value_rows) || !valid_output(value, key_value_rows)) {
        throw std::invalid_argument("block-FP8 attention projection: invalid output tensors");
    }
    const AttentionOutput output{static_cast<__nv_bfloat16*>(query.data),
                                 static_cast<__nv_bfloat16*>(gate.data),
                                 static_cast<__nv_bfloat16*>(key.data),
                                 static_cast<__nv_bfloat16*>(value.data), weight.n};
    if (marlin) {
        if (!launch_marlin_leaf(x, weight, output, stream)) {
            launch<AttentionOutput, true>(x, weight, output, stream);
        }
        return;
    }
    if (use_hmma(x, weight, Fp8BlockOperation::Attention)) {
        launch_hmma(x, weight, output, stream);
    } else {
        launch(x, weight, output, stream);
    }
}

void fp8_block_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
    const bool marlin = validate_projection_input(x, weight, "block-FP8 GDN projection");
    if ((weight.n != 16384 && weight.n != 8192) || weight.k != 5120) {
        throw std::invalid_argument("block-FP8 GDN projection: unsupported weight shape");
    }
    const int qkv_rows = weight.n == 16384 ? 10240 : 5120;
    const int z_rows = weight.n - qkv_rows;
    const auto valid_output = [&](const Tensor& output, int rows) {
        return output.dtype == DType::BF16 && output.ne[0] == rows && output.ne[1] == x.ne[1] &&
               output.ne[2] == 1 && output.ne[3] == 1 && output.is_contiguous() &&
               output.data != nullptr && (reinterpret_cast<std::uintptr_t>(output.data) & 15U) == 0;
    };
    if (!valid_output(qkv, qkv_rows) || !valid_output(z, z_rows)) {
        throw std::invalid_argument("block-FP8 GDN projection: invalid output tensors");
    }
    const GdnOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                           static_cast<__nv_bfloat16*>(z.data), weight.n};
    if (marlin) {
        if (!launch_marlin_leaf(x, weight, output, stream)) {
            launch<GdnOutput, true>(x, weight, output, stream);
        }
        return;
    }
    if (use_hmma(x, weight, Fp8BlockOperation::Gdn)) {
        launch_hmma(x, weight, output, stream);
    } else {
        launch(x, weight, output, stream);
    }
}

} // namespace ninfer::ops::detail