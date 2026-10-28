#include "ops/ops.h"

#include <cmath>

namespace ops {

void matvec(float* out, const float* W, const float* x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* w = W + size_t(r) * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) acc += w[c] * x[c];
        out[r] = acc;
    }
}

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
}

void rope(float* x, int n_heads, int head_dim, int pos, float theta) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        // Match HF: inv_freq is computed in fp32, then multiplied by the position.
        float inv_freq = 1.0f / std::pow(theta, float(2 * i) / float(head_dim));
        float angle = float(pos) * inv_freq;
        float c = std::cos(angle), s = std::sin(angle);
        for (int h = 0; h < n_heads; ++h) {
            float* v = x + h * head_dim;
            float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
}

void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = x[i] > mx ? x[i] : mx;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

void swiglu(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) {
        float g = a[i];
        a[i] = g / (1.0f + std::exp(-g)) * b[i];
    }
}

void add(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) a[i] += b[i];
}

}  // namespace ops
