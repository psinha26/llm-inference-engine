// Picks the next token from logits: greedy, or temperature + top-k + top-p sampling.
#pragma once

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

struct SamplerConfig {
    float temperature = 0.0f;  // 0 = greedy (argmax)
    int top_k = 0;             // 0 = disabled
    float top_p = 1.0f;        // 1 = disabled
    uint64_t seed = 42;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg) : cfg_(cfg), rng_(cfg.seed) {}
    int sample(const float* logits, int vocab);

private:
    SamplerConfig cfg_;
    std::mt19937_64 rng_;
    std::vector<std::pair<float, int>> cand_;  // reused scratch: (prob, id)
};
