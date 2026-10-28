// Matrix-vector kernels: the hot loop of batch-1 inference (>95% of the FLOPs).
//
// Every kernel computes rows [r0, r1) of out = W . x so the thread pool can split rows.
// Implementations, in speedup-ladder order:
//   Naive   - plain scalar loops (the reference)
//   Simd    - NEON (Apple Silicon / ARM) or AVX2+FMA (x86) intrinsics, one row at a time
//   Blocked - SIMD over 4 rows at once: each x vector loaded from cache feeds 4 rows of FMAs.
//             On a row-major matrix this reads 4 separate streams, one per row.
//
// Tiled layout (scripts/convert.py --layout tiled): rows are grouped in 4s (padded with zero
// rows) and interleaved in small tiles, so the blocked kernel reads ONE sequential stream:
//   fp32: [row group][8-column chunk][4 rows x 8 values]
//   int8: [row group][32-column block][4 rows x 32 int8], scales [row group][block][4 rows]
//
// int8 ("q8") weights use blocks of 32 values sharing one fp32 scale:  w[i] ~= d * q[i].
// The activation vector is quantized the same way, so each block's dot product is a pure
// int8 x int8 -> int32 sum, scaled once by d_w * d_x.
#pragma once

#include <cstdint>

namespace kernels {

constexpr int kQ8Block = 32;

enum class Impl { Naive, Simd, Blocked };

// True if this CPU/build has a SIMD backend; name is "NEON", "NEON+dotprod", "AVX2" or "none".
bool simd_supported();
const char* simd_name();

void matvec_f32(Impl impl, float* out, const float* W, const float* x, int r0, int r1, int cols);

// W: int8 values [rows, cols] + scales [rows, cols/32]; x: int8 [cols] + scales [cols/32].
void matvec_q8(Impl impl, float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int r0,
               int r1, int cols);

// Same, for matrices stored in the tiled layout (always uses the 4-row blocked kernels).
// `rows` is the true row count; output rows >= rows are never written.
void matvec_f32_tiled(Impl impl, float* out, const float* W, const float* x, int r0, int r1, int rows, int cols);
void matvec_q8_tiled(Impl impl, float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd,
                     int r0, int r1, int rows, int cols);

constexpr int kTileRows = 4;
constexpr int kTileColsF32 = 8;

// Quantize n floats (n % 32 == 0) into int8 blocks with one scale per 32 values.
void quantize_q8(const float* x, int n, int8_t* q, float* d);

}  // namespace kernels
