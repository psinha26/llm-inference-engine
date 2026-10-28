// NEON kernels for AArch64 (Apple Silicon, ARM Linux). The int8 path uses the SDOT
// instruction (vdotq_s32) when the build targets ARMv8.2 dotprod, else widening multiplies.
#if defined(__aarch64__)

#include <arm_neon.h>

#include <cstddef>

#include "ops/kernels.h"
#include "ops/kernels_internal.h"

#if defined(__linux__) && defined(__ARM_FEATURE_DOTPROD)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif

namespace kernels::backend {
namespace {

inline float dot_row(const float* w, const float* x, int cols) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;  // 4 chains hide FMA latency
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(w + c), vld1q_f32(x + c));
        a1 = vfmaq_f32(a1, vld1q_f32(w + c + 4), vld1q_f32(x + c + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(w + c + 8), vld1q_f32(x + c + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(w + c + 12), vld1q_f32(x + c + 12));
    }
    for (; c + 4 <= cols; c += 4) a0 = vfmaq_f32(a0, vld1q_f32(w + c), vld1q_f32(x + c));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
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
        float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const float32x4_t x0 = vld1q_f32(x + c), x1 = vld1q_f32(x + c + 4);  // loaded once, used 4x
            a00 = vfmaq_f32(a00, vld1q_f32(w0 + c), x0);
            a01 = vfmaq_f32(a01, vld1q_f32(w0 + c + 4), x1);
            a10 = vfmaq_f32(a10, vld1q_f32(w1 + c), x0);
            a11 = vfmaq_f32(a11, vld1q_f32(w1 + c + 4), x1);
            a20 = vfmaq_f32(a20, vld1q_f32(w2 + c), x0);
            a21 = vfmaq_f32(a21, vld1q_f32(w2 + c + 4), x1);
            a30 = vfmaq_f32(a30, vld1q_f32(w3 + c), x0);
            a31 = vfmaq_f32(a31, vld1q_f32(w3 + c + 4), x1);
        }
        float s0 = vaddvq_f32(vaddq_f32(a00, a01)), s1 = vaddvq_f32(vaddq_f32(a10, a11));
        float s2 = vaddvq_f32(vaddq_f32(a20, a21)), s3 = vaddvq_f32(vaddq_f32(a30, a31));
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

// Dot product of one 32-value int8 block, as 4 int32 lanes.
inline int32x4_t dot_block(const int8_t* w, const int8x16_t x0, const int8x16_t x1) {
    const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16);
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
#else
    // |product| <= 127*127, so a pair summed into int16 cannot overflow.
    int16x8_t p = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
    p = vmlal_s8(p, vget_high_s8(w0), vget_high_s8(x0));
    int16x8_t p2 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
    p2 = vmlal_s8(p2, vget_high_s8(w1), vget_high_s8(x1));
    return vpadalq_s16(vpaddlq_s16(p), p2);
#endif
}

void q8_simd(float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int r0, int r1,
             int cols) {
    const int nb = cols / kQ8Block;
    for (int r = r0; r < r1; ++r) {
        const int8_t* q = Wq + size_t(r) * cols;
        const float* d = Wd + size_t(r) * nb;
        float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; ++b) {
            const int8x16_t x0 = vld1q_s8(xq + b * kQ8Block), x1 = vld1q_s8(xq + b * kQ8Block + 16);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(dot_block(q + b * kQ8Block, x0, x1)), d[b] * xd[b]);
        }
        out[r] = vaddvq_f32(acc);
    }
}

void q8_blocked(float* out, const int8_t* Wq, const float* Wd, const int8_t* xq, const float* xd, int r0, int r1,
                int cols) {
    const int nb = cols / kQ8Block;
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        const int8_t* q0 = Wq + size_t(r) * cols;
        const float* d0 = Wd + size_t(r) * nb;
        for (int b = 0; b < nb; ++b) {
            const int8x16_t x0 = vld1q_s8(xq + b * kQ8Block), x1 = vld1q_s8(xq + b * kQ8Block + 16);
            const float dx = xd[b];
            const int off = b * kQ8Block;
            acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(dot_block(q0 + off, x0, x1)), d0[b] * dx);
            acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(dot_block(q0 + cols + off, x0, x1)), d0[nb + b] * dx);
            acc2 = vmlaq_n_f32(acc2, vcvtq_f32_s32(dot_block(q0 + 2 * size_t(cols) + off, x0, x1)), d0[2 * nb + b] * dx);
            acc3 = vmlaq_n_f32(acc3, vcvtq_f32_s32(dot_block(q0 + 3 * size_t(cols) + off, x0, x1)), d0[3 * nb + b] * dx);
        }
        out[r] = vaddvq_f32(acc0);
        out[r + 1] = vaddvq_f32(acc1);
        out[r + 2] = vaddvq_f32(acc2);
        out[r + 3] = vaddvq_f32(acc3);
    }
    q8_simd(out, Wq, Wd, xq, xd, r, r1, cols);
}

// Tiled fp32: one sequential stream of [4 rows x 8 values] tiles per row group.
void f32_tiled(float* out, const float* W, const float* x, int g0, int g1, int rows, int cols) {
    for (int g = g0; g < g1; ++g) {
        const float* p = W + size_t(g) * 4 * cols;
        float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        for (int c = 0; c < cols; c += 8, p += 32) {
            const float32x4_t x0 = vld1q_f32(x + c), x1 = vld1q_f32(x + c + 4);
            a00 = vfmaq_f32(a00, vld1q_f32(p), x0);
            a01 = vfmaq_f32(a01, vld1q_f32(p + 4), x1);
            a10 = vfmaq_f32(a10, vld1q_f32(p + 8), x0);
            a11 = vfmaq_f32(a11, vld1q_f32(p + 12), x1);
            a20 = vfmaq_f32(a20, vld1q_f32(p + 16), x0);
            a21 = vfmaq_f32(a21, vld1q_f32(p + 20), x1);
            a30 = vfmaq_f32(a30, vld1q_f32(p + 24), x0);
            a31 = vfmaq_f32(a31, vld1q_f32(p + 28), x1);
        }
        const float s[4] = {vaddvq_f32(vaddq_f32(a00, a01)), vaddvq_f32(vaddq_f32(a10, a11)),
                            vaddvq_f32(vaddq_f32(a20, a21)), vaddvq_f32(vaddq_f32(a30, a31))};
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
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        for (int b = 0; b < nb; ++b, q += 128, d += 4) {
            const int8x16_t x0 = vld1q_s8(xq + b * kQ8Block), x1 = vld1q_s8(xq + b * kQ8Block + 16);
            const float dx = xd[b];
            acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(dot_block(q, x0, x1)), d[0] * dx);
            acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(dot_block(q + 32, x0, x1)), d[1] * dx);
            acc2 = vmlaq_n_f32(acc2, vcvtq_f32_s32(dot_block(q + 64, x0, x1)), d[2] * dx);
            acc3 = vmlaq_n_f32(acc3, vcvtq_f32_s32(dot_block(q + 96, x0, x1)), d[3] * dx);
        }
        const float s[4] = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
        store_group(out, g, rows, s);
    }
}

#if defined(__ARM_FEATURE_DOTPROD)
const Backend kNeon = {"NEON+dotprod", f32_simd, f32_blocked, q8_simd, q8_blocked, f32_tiled, q8_tiled};
#else
const Backend kNeon = {"NEON", f32_simd, f32_blocked, q8_simd, q8_blocked, f32_tiled, q8_tiled};
#endif

}  // namespace

const Backend* neon() {
#if defined(__linux__) && defined(__ARM_FEATURE_DOTPROD)
    if (!(getauxval(AT_HWCAP) & HWCAP_ASIMDDP)) return nullptr;  // built for dotprod, CPU lacks it
#endif
    return &kNeon;
}

}  // namespace kernels::backend

#endif  // __aarch64__
