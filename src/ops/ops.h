// Transformer building blocks. These are the plain-loop reference versions: every
// optimisation later in the speedup ladder must produce the same numbers as these.
#pragma once

#include <cstddef>

namespace ops {

// out[rows] = W[rows, cols] . x[cols]   (W row-major, as stored in the checkpoint)
void matvec(float* out, const float* W, const float* x, int rows, int cols);

// out = x / sqrt(mean(x^2) + eps) * weight
void rmsnorm(float* out, const float* x, const float* weight, int n, float eps);

// Rotary position embedding, "rotate-half" layout (HF Llama): for each head the pairs are
// (x[i], x[i + head_dim/2]), rotated by angle pos * theta^(-2i/head_dim).
void rope(float* x, int n_heads, int head_dim, int pos, float theta);

// In-place numerically stable softmax over n values.
void softmax(float* x, int n);

// a = silu(a) * b   (SwiGLU gate; silu(x) = x * sigmoid(x))
void swiglu(float* a, const float* b, int n);

// a += b
void add(float* a, const float* b, int n);

}  // namespace ops
