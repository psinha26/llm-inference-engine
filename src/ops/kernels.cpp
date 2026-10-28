#include "ops/kernels.h"

#include <cmath>

#include "ops/kernels_internal.h"

namespace kernels {
namespace backend {

void f32_naive(float* out, const float* W, const float* x, int r0, int r1, int cols) {
    for (int r = r0; r < r1; ++r) {
        const float* w = W + size_t(r) * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) acc += w[c] * x[c];
        out[r] = acc;
    }
}

void q8_naive(float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int r0, int r1,
              int cols) {
    const int nb = cols / kQ8Block;
    for (int r = r0; r < r1; ++r) {
        const int8_t* q = Wq + size_t(r) * cols;
        const float* d = Wd + size_t(r) * nb;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b) {
            int32_t isum = 0;
            for (int i = 0; i < kQ8Block; ++i) isum += int32_t(q[b * kQ8Block + i]) * int32_t(xq[b * kQ8Block + i]);
            acc += float(isum) * d[b] * xd[b];
        }
        out[r] = acc;
    }
}

void f32_tiled_naive(float* out, const float* W, const float* x, int g0, int g1, int rows, int cols) {
    for (int g = g0; g < g1; ++g) {
        const float* p = W + size_t(g) * kTileRows * cols;
        float s[4] = {0, 0, 0, 0};
        for (int c = 0; c < cols; c += kTileColsF32, p += kTileRows * kTileColsF32)
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < kTileColsF32; ++i) s[j] += p[j * kTileColsF32 + i] * x[c + i];
        store_group(out, g, rows, s);
    }
}

void q8_tiled_naive(float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int g0, int g1,
                    int rows, int cols) {
    const int nb = cols / kQ8Block;
    for (int g = g0; g < g1; ++g) {
        const int8_t* q = Wq + size_t(g) * kTileRows * cols;
        const float* d = Wd + size_t(g) * kTileRows * nb;
        float s[4] = {0, 0, 0, 0};
        for (int b = 0; b < nb; ++b, q += kTileRows * kQ8Block, d += kTileRows)
            for (int j = 0; j < 4; ++j) {
                int32_t isum = 0;
                for (int i = 0; i < kQ8Block; ++i) isum += int32_t(q[j * kQ8Block + i]) * xq[b * kQ8Block + i];
                s[j] += float(isum) * d[j] * xd[b];
            }
        store_group(out, g, rows, s);
    }
}

}  // namespace backend

namespace {

const backend::Backend* detect() {
#if defined(__aarch64__)
    return backend::neon();
#elif defined(__x86_64__)
    return backend::avx2();
#else
    return nullptr;
#endif
}

const backend::Backend* simd() {
    static const backend::Backend* b = detect();
    return b;
}

}  // namespace

bool simd_supported() { return simd() != nullptr; }
const char* simd_name() { return simd() ? simd()->name : "none"; }

void matvec_f32(Impl impl, float* out, const float* W, const float* x, int r0, int r1, int cols) {
    const backend::Backend* b = simd();
    if (impl == Impl::Naive || !b) return backend::f32_naive(out, W, x, r0, r1, cols);
    if (impl == Impl::Simd) return b->f32_simd(out, W, x, r0, r1, cols);
    return b->f32_blocked(out, W, x, r0, r1, cols);
}

void matvec_q8(Impl impl, float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int r0,
               int r1, int cols) {
    const backend::Backend* b = simd();
    if (impl == Impl::Naive || !b) return backend::q8_naive(out, Wq, Wd, xq, xd, r0, r1, cols);
    if (impl == Impl::Simd) return b->q8_simd(out, Wq, Wd, xq, xd, r0, r1, cols);
    return b->q8_blocked(out, Wq, Wd, xq, xd, r0, r1, cols);
}

// Row ranges from the thread pool are multiples of 4, i.e. whole row groups.
void matvec_f32_tiled(Impl impl, float* out, const float* W, const float* x, int r0, int r1, int rows, int cols) {
    const backend::Backend* b = simd();
    const int g0 = r0 / kTileRows, g1 = (r1 + kTileRows - 1) / kTileRows;
    if (impl == Impl::Naive || !b) return backend::f32_tiled_naive(out, W, x, g0, g1, rows, cols);
    b->f32_tiled(out, W, x, g0, g1, rows, cols);
}

void matvec_q8_tiled(Impl impl, float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd,
                     int r0, int r1, int rows, int cols) {
    const backend::Backend* b = simd();
    const int g0 = r0 / kTileRows, g1 = (r1 + kTileRows - 1) / kTileRows;
    if (impl == Impl::Naive || !b) return backend::q8_tiled_naive(out, Wq, Wd, xq, xd, g0, g1, rows, cols);
    b->q8_tiled(out, Wq, Wd, xq, xd, g0, g1, rows, cols);
}

void quantize_q8(const float* x, int n, int8_t* q, float* d) {
    for (int b = 0; b < n / kQ8Block; ++b) {
        const float* xb = x + b * kQ8Block;
        float amax = 0.0f;
        for (int i = 0; i < kQ8Block; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
        const float scale = amax / 127.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        d[b] = scale;
        for (int i = 0; i < kQ8Block; ++i) q[b * kQ8Block + i] = int8_t(std::lrintf(xb[i] * inv));
    }
}

}  // namespace kernels
