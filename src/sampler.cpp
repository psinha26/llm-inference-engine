#include "sampler.h"

#include <algorithm>
#include <cmath>

int Sampler::sample(const float* logits, int vocab) {
    if (cfg_.temperature <= 0.0f) {
        return int(std::max_element(logits, logits + vocab) - logits);
    }

    cand_.resize(vocab);
    for (int i = 0; i < vocab; ++i) cand_[i] = {logits[i] / cfg_.temperature, i};
    auto by_logit_desc = [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };

    // Top-k: keep only the k highest logits (partial sort is O(V log k)).
    size_t n = cand_.size();
    if (cfg_.top_k > 0 && size_t(cfg_.top_k) < n) {
        std::partial_sort(cand_.begin(), cand_.begin() + cfg_.top_k, cand_.end(), by_logit_desc);
        n = size_t(cfg_.top_k);
    } else {
        std::sort(cand_.begin(), cand_.end(), by_logit_desc);
    }

    // Softmax over the survivors (they're sorted, so cand_[0] holds the max).
    float mx = cand_[0].first, sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        cand_[i].first = std::exp(cand_[i].first - mx);
        sum += cand_[i].first;
    }
    for (size_t i = 0; i < n; ++i) cand_[i].first /= sum;

    // Top-p (nucleus): smallest prefix whose probability mass reaches top_p.
    if (cfg_.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cut = n;
        for (size_t i = 0; i < n; ++i) {
            cum += cand_[i].first;
            if (cum >= cfg_.top_p) { cut = i + 1; break; }
        }
        n = cut;
        sum = cum;
    } else {
        sum = 1.0f;
    }

    std::uniform_real_distribution<float> uni(0.0f, sum);
    float r = uni(rng_), acc = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        acc += cand_[i].first;
        if (r < acc) return cand_[i].second;
    }
    return cand_[n - 1].second;
}
