// Backend entry points; each backend .cpp is compiled only for its architecture.
#pragma once

#include <cstdint>

namespace kernels::backend {

using F32Fn = void (*)(float*, const float*, const float*, int, int, int);
using Q8Fn = void (*)(float*, const int8_t*, const float*, const int8_t*, const float*, int, int, int);
// Tiled kernels take (out, W, x, g0, g1, rows, cols): row groups [g0, g1), writing rows < rows.
using F32TiledFn = void (*)(float*, const float*, const float*, int, int, int, int);
using Q8TiledFn = void (*)(float*, const int8_t*, const float*, const int8_t*, const float*, int, int, int, int);

struct Backend {
    const char* name;
    F32Fn f32_simd, f32_blocked;
    Q8Fn q8_simd, q8_blocked;
    F32TiledFn f32_tiled;
    Q8TiledFn q8_tiled;
};

// Store a computed 4-row group, dropping padding rows past the end of the matrix.
inline void store_group(float* out, int g, int rows, const float s[4]) {
    for (int j = 0; j < 4; ++j)
        if (g * 4 + j < rows) out[g * 4 + j] = s[j];
}

// Scalar reference kernels.
void f32_naive(float* out, const float* W, const float* x, int r0, int r1, int cols);
void q8_naive(float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int r0, int r1,
              int cols);
void f32_tiled_naive(float* out, const float* W, const float* x, int g0, int g1, int rows, int cols);
void q8_tiled_naive(float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int g0, int g1,
                    int rows, int cols);

#if defined(__aarch64__)
const Backend* neon();
#endif
#if defined(__x86_64__)
const Backend* avx2();  // nullptr if the CPU lacks AVX2/FMA
#endif

}  // namespace kernels::backend
