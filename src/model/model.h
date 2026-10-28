// Llama-family transformer: weight loading (mmap) and the forward pass.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// On-disk header written by scripts/convert.py. Keep the two in sync.
constexpr uint32_t kMagic = 0x454D4C4C;  // "LLME"
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderBytes = 256;
constexpr size_t kTensorAlign = 64;
constexpr int32_t kFlagTied = 1 << 0;
constexpr int32_t kFlagQkvBias = 1 << 1;

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
};

struct LayerWeights {
    const float* attn_norm;
    const float *wq, *wk, *wv;
    const float *bq = nullptr, *bk = nullptr, *bv = nullptr;
    const float* wo;
    const float* mlp_norm;
    const float *w_gate, *w_up, *w_down;
};

struct Weights {
    const float* token_embedding;
    std::vector<LayerWeights> layers;
    const float* final_norm;
    const float* lm_head;  // aliases token_embedding when tied
};

// Activation buffers and the KV cache, preallocated once for a fixed context length.
struct RunState {
    std::vector<float> x, xb, xb2;  // residual stream and scratch [dim]
    std::vector<float> q;           // [q_dim]
    std::vector<float> hb, hb2;     // MLP scratch [hidden_dim]
    std::vector<float> att;         // attention scores [n_heads * ctx]
    std::vector<float> logits;      // [vocab]
    std::vector<float> key_cache;   // [n_layers, ctx, kv_dim]
    std::vector<float> value_cache; // [n_layers, ctx, kv_dim]
};

class Model {
public:
    Model(const std::string& path, int ctx_len);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Config& config() const { return cfg_; }
    int ctx_len() const { return ctx_; }

    // Run one token at position `pos`, appending its keys/values to the cache.
    // Positions 0..pos-1 must already be in the cache. Returns logits [vocab] when
    // compute_logits is set (skipped during prefill except for the last prompt token).
    const float* forward(int token, int pos, bool compute_logits = true);

private:
    Config cfg_;
    Weights w_;
    RunState s_;
    int ctx_ = 0;
    void* map_ = nullptr;
    size_t map_size_ = 0;
};
