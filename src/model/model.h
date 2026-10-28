// Llama-family transformer: weight loading (mmap) and the forward pass.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "ops/kernels.h"
#include "ops/thread_pool.h"

// On-disk header written by scripts/convert.py. Keep the two in sync.
constexpr uint32_t kMagic = 0x454D4C4C;  // "LLME"
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderBytes = 256;
constexpr size_t kTensorAlign = 64;
constexpr int32_t kFlagTied = 1 << 0;
constexpr int32_t kFlagQkvBias = 1 << 1;
constexpr int32_t kFlagQ8 = 1 << 2;     // weight matrices stored as int8 blocks + fp32 scales
constexpr int32_t kFlagTiled = 1 << 3;  // weight matrices stored in the 4-row tiled layout

struct Config {
    int dim = 0;          // hidden size
    int hidden_dim = 0;   // MLP intermediate size
    int n_layers = 0;
    int n_heads = 0;      // query heads
    int n_kv_heads = 0;   // key/value heads (grouped-query attention when < n_heads)
    int vocab_size = 0;
    int max_seq_len = 0;
    int head_dim = 0;
    int32_t flags = 0;
    float rope_theta = 10000.0f;
    float norm_eps = 1e-5f;

    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
    bool q8() const { return flags & kFlagQ8; }
    bool tiled() const { return flags & kFlagTiled; }
};

// A weight matrix [rows, cols], either fp32 or int8 blocks (q + one scale d per 32 values),
// in row-major or 4-row tiled layout (see ops/kernels.h).
struct Mat {
    const float* f = nullptr;
    const int8_t* q = nullptr;
    const float* d = nullptr;
    int rows = 0, cols = 0;
    bool tiled = false;

    // Copy row r out as fp32 (dequantizing / un-tiling as needed). Used for embedding lookup.
    void row(int r, float* out) const;
};

struct LayerWeights {
    const float* attn_norm;
    Mat wq, wk, wv;
    const float *bq = nullptr, *bk = nullptr, *bv = nullptr;
    Mat wo;
    const float* mlp_norm;
    Mat w_gate, w_up, w_down;
};

struct Weights {
    Mat token_embedding;
    std::vector<LayerWeights> layers;
    const float* final_norm;
    Mat lm_head;  // aliases token_embedding when tied
};

// 64-byte aligned allocator: buffers start on a cache line, so SIMD loads never straddle two.
template <class T>
struct AlignedAlloc {
    using value_type = T;
    AlignedAlloc() = default;
    template <class U>
    AlignedAlloc(const AlignedAlloc<U>&) {}
    T* allocate(size_t n) {
        void* p = std::aligned_alloc(64, (n * sizeof(T) + 63) / 64 * 64);
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) { std::free(p); }
    template <class U>
    bool operator==(const AlignedAlloc<U>&) const { return true; }
    template <class U>
    bool operator!=(const AlignedAlloc<U>&) const { return false; }
};
template <class T>
using avec = std::vector<T, AlignedAlloc<T>>;

// Activation buffers and the KV cache, preallocated once for a fixed context length.
struct RunState {
    avec<float> x, xb, xb2;      // residual stream and scratch [dim]
    avec<float> q;               // [q_dim]
    avec<float> hb, hb2;         // MLP scratch [hidden_dim]
    avec<float> att;             // attention scores [n_heads * ctx]
    avec<float> logits;          // [vocab]
    avec<float> key_cache;       // [n_layers, ctx, kv_dim]
    avec<float> value_cache;     // [n_layers, ctx, kv_dim]
    avec<int8_t> xq;             // int8-quantized activation (q8 models)
    avec<float> xd;              // its block scales
};

struct ModelOptions {
    int threads = 1;
    kernels::Impl impl = kernels::Impl::Blocked;
    // false: mmap the file (instant start, pages loaded lazily, shared with the page cache).
    // true:  read it into one allocation backed by huge pages where the OS supports them, so
    //        streaming the weights each token causes far fewer TLB misses.
    bool no_mmap = false;
};

class Model {
public:
    Model(const std::string& path, int ctx_len, const ModelOptions& opts = {});
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Config& config() const { return cfg_; }
    int ctx_len() const { return ctx_; }
    size_t weight_bytes() const { return map_size_; }

    // Run one token at position `pos`, appending its keys/values to the cache.
    // Positions 0..pos-1 must already be in the cache. Returns logits [vocab] when
    // compute_logits is set (skipped during prefill except for the last prompt token).
    const float* forward(int token, int pos, bool compute_logits = true);

private:
    struct MatOut {
        float* out;
        const Mat* W;
    };
    void matmul(std::initializer_list<MatOut> jobs, const float* x);
    void attention(int layer, int pos);

    Config cfg_;
    Weights w_;
    RunState s_;
    ModelOptions opts_;
    std::unique_ptr<ThreadPool> pool_;
    int ctx_ = 0;
    void* map_ = nullptr;
    size_t map_size_ = 0;
    bool owned_ = false;  // map_ is a heap allocation (no_mmap) rather than an mmap
};
