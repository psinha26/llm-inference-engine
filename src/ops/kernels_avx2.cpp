// AVX2 + FMA kernels for x86-64. Compiled with -mavx2 -mfma; only used after a CPUID check.
#if defined(__x86_64__)

#include <immintrin.h>

#include <cstddef>

#include "ops/kernels.h"
#include "ops/kernels_internal.h"

namespace kernels::backend {
namespace {

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline float dot_row(const float* w, const float* x, int cols) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;  // 4 chains hide FMA latency
    int c = 0;
    for (; c + 32 <= cols; c += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c), _mm256_loadu_ps(x + c), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c + 8), _mm256_loadu_ps(x + c + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c + 16), _mm256_loadu_ps(x + c + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c + 24), _mm256_loadu_ps(x + c + 24), a3);
    }
    for (; c + 8 <= cols; c += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c), _mm256_loadu_ps(x + c), a0);
    float s = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; c < cols; ++c) s += w[c] * x[c];
    return s;
}

void f32_simd(float* out, const float* W, const float* x, int r0, int r1, int cols) {
    for (int r = r0; r < r1; ++r) out[r] = dot_row(W + size_t(r) * cols, x, cols);
}

void f32_blocked(float* out, const float* W, const float* x, int r0, int r1, int cols) {
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const float* w0 = W + size_t(r) * cols;
        const float* w1 = w0 + cols;
        const float* w2 = w1 + cols;
        const float* w3 = w2 + cols;
        __m256 a00 = _mm256_setzero_ps(), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        int c = 0;
        for (; c + 16 <= cols; c += 16) {
            const __m256 x0 = _mm256_loadu_ps(x + c), x1 = _mm256_loadu_ps(x + c + 8);  // loaded once, used 4x
            a00 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + c), x0, a00);
            a01 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + c + 8), x1, a01);
            a10 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + c), x0, a10);
            a11 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + c + 8), x1, a11);
            a20 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + c), x0, a20);
            a21 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + c + 8), x1, a21);
            a30 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + c), x0, a30);
            a31 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + c + 8), x1, a31);
        }
        float s0 = hsum(_mm256_add_ps(a00, a01)), s1 = hsum(_mm256_add_ps(a10, a11));
        float s2 = hsum(_mm256_add_ps(a20, a21)), s3 = hsum(_mm256_add_ps(a30, a31));
        for (; c < cols; ++c) {
            s0 += w0[c] * x[c];
            s1 += w1[c] * x[c];
            s2 += w2[c] * x[c];
            s3 += w3[c] * x[c];
        }
        out[r] = s0;
        out[r + 1] = s1;
        out[r + 2] = s2;
        out[r + 3] = s3;
    }
    f32_simd(out, W, x, r, r1, cols);
}

// int8 x int8 dot product of one 32-value block, as 8 int32 lanes.
// maddubs needs unsigned x signed, so move w's sign onto x: |w| * (x * sign(w)) = w * x.
inline __m256i dot_block(__m256i w, __m256i x) {
    const __m256i aw = _mm256_sign_epi8(w, w);
    const __m256i sx = _mm256_sign_epi8(x, w);
    const __m256i p16 = _mm256_maddubs_epi16(aw, sx);  // max 2*127*127 = 32258, no saturation
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}

void q8_simd(float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int r0, int r1,
             int cols) {
    const int nb = cols / kQ8Block;
    for (int r = r0; r < r1; ++r) {
        const int8_t* q = Wq + size_t(r) * cols;
        const float* d = Wd + size_t(r) * nb;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + b * kQ8Block));
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + b * kQ8Block));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_block(w, x)), _mm256_set1_ps(d[b] * xd[b]), acc);
        }
        out[r] = hsum(acc);
    }
}

void q8_blocked(float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int r0, int r1,
                int cols) {
    const int nb = cols / kQ8Block;
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int b = 0; b < nb; ++b) {
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + b * kQ8Block));
            for (int k = 0; k < 4; ++k) {
                const size_t row = size_t(r + k);
                const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Wq + row * cols + b * kQ8Block));
                acc[k] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_block(w, x)), _mm256_set1_ps(Wd[row * nb + b] * xd[b]),
                                         acc[k]);
            }
        }
        for (int k = 0; k < 4; ++k) out[r + k] = hsum(acc[k]);
    }
    q8_simd(out, Wq, Wd, xq, xd, r, r1, cols);
}

// Tiled fp32: one sequential stream of [4 rows x 8 values] tiles per row group.
// Two tiles per iteration gives 8 independent FMA chains.
void f32_tiled(float* out, const float* W, const float* x, int g0, int g1, int rows, int cols) {
    for (int g = g0; g < g1; ++g) {
        const float* p = W + size_t(g) * 4 * cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        int c = 0;
        for (; c + 16 <= cols; c += 16, p += 64) {
            const __m256 x0 = _mm256_loadu_ps(x + c), x1 = _mm256_loadu_ps(x + c + 8);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p), x0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 8), x0, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 16), x0, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 24), x0, a3);
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 32), x1, b0);
            b1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 40), x1, b1);
            b2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 48), x1, b2);
            b3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 56), x1, b3);
        }
        for (; c < cols; c += 8, p += 32) {
            const __m256 x0 = _mm256_loadu_ps(x + c);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p), x0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 8), x0, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 16), x0, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 24), x0, a3);
        }
        const float s[4] = {hsum(_mm256_add_ps(a0, b0)), hsum(_mm256_add_ps(a1, b1)), hsum(_mm256_add_ps(a2, b2)),
                            hsum(_mm256_add_ps(a3, b3))};
        store_group(out, g, rows, s);
    }
}

// Tiled int8: per 32-column block, 4 rows x 32 int8 contiguous, then 4 row scales.
void q8_tiled(float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int g0, int g1,
              int rows, int cols) {
    const int nb = cols / kQ8Block;
    for (int g = g0; g < g1; ++g) {
        const int8_t* q = Wq + size_t(g) * 4 * cols;
        const float* d = Wd + size_t(g) * 4 * nb;
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int b = 0; b < nb; ++b, q += 128, d += 4) {
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + b * kQ8Block));
            const float dx = xd[b];
            for (int j = 0; j < 4; ++j) {
                const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32 * j));
                acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_block(w, x)), _mm256_set1_ps(d[j] * dx), acc[j]);
            }
        }
        const float s[4] = {hsum(acc[0]), hsum(acc[1]), hsum(acc[2]), hsum(acc[3])};
        store_group(out, g, rows, s);
    }
}

const Backend kAvx2 = {"AVX2", f32_simd, f32_blocked, q8_simd, q8_blocked, f32_tiled, q8_tiled};

}  // namespace

const Backend* avx2() {
    __builtin_cpu_init();
    return (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? &kAvx2 : nullptr;
}

}  // namespace kernels::backend

#endif  // __x86_64__
