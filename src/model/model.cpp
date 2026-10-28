#include "model/model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "ops/ops.h"

namespace {

// Walks the tensor section of the file, handing out aligned pointers in write order.
class TensorCursor {
public:
    TensorCursor(const char* base, size_t size) : base_(base), size_(size), off_(kHeaderBytes) {}
    const float* next(size_t n_floats) {
        off_ = (off_ + kTensorAlign - 1) / kTensorAlign * kTensorAlign;
        size_t bytes = n_floats * sizeof(float);
        if (off_ + bytes > size_) throw std::runtime_error("model file is truncated");
        const float* p = reinterpret_cast<const float*>(base_ + off_);
        off_ += bytes;
        return p;
    }
    size_t offset() const { return off_; }

private:
    const char* base_;
    size_t size_, off_;
};

}  // namespace

Model::Model(const std::string& path, int ctx_len) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open model file " + path);
    struct stat st;
    fstat(fd, &st);
    map_size_ = size_t(st.st_size);
    map_ = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_ == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
    if (map_size_ < kHeaderBytes) throw std::runtime_error("model file too small");

    const char* base = static_cast<const char*>(map_);
    uint32_t magic, version;
    std::memcpy(&magic, base, 4);
    std::memcpy(&version, base + 4, 4);
    if (magic != kMagic) throw std::runtime_error("not an engine model file (bad magic); run scripts/convert.py");
    if (version != kVersion) throw std::runtime_error("unsupported model file version");
    int32_t ints[9];
    std::memcpy(ints, base + 8, sizeof(ints));
    cfg_.dim = ints[0];
    cfg_.hidden_dim = ints[1];
    cfg_.n_layers = ints[2];
    cfg_.n_heads = ints[3];
    cfg_.n_kv_heads = ints[4];
    cfg_.vocab_size = ints[5];
    cfg_.max_seq_len = ints[6];
    cfg_.head_dim = ints[7];
    cfg_.flags = ints[8];
    std::memcpy(&cfg_.rope_theta, base + 44, 4);
    std::memcpy(&cfg_.norm_eps, base + 48, 4);
    if (cfg_.n_heads % cfg_.n_kv_heads != 0) throw std::runtime_error("n_heads must be a multiple of n_kv_heads");

    const Config& c = cfg_;
    const size_t D = c.dim, H = c.hidden_dim, Q = c.q_dim(), KV = c.kv_dim(), V = c.vocab_size;
    TensorCursor cur(base, map_size_);
    w_.token_embedding = cur.next(V * D);
    w_.layers.resize(c.n_layers);
    for (auto& L : w_.layers) {
        L.attn_norm = cur.next(D);
        L.wq = cur.next(Q * D);
        L.wk = cur.next(KV * D);
        L.wv = cur.next(KV * D);
        if (c.flags & kFlagQkvBias) {
            L.bq = cur.next(Q);
            L.bk = cur.next(KV);
            L.bv = cur.next(KV);
        }
        L.wo = cur.next(D * Q);
        L.mlp_norm = cur.next(D);
        L.w_gate = cur.next(H * D);
        L.w_up = cur.next(H * D);
        L.w_down = cur.next(D * H);
    }
    w_.final_norm = cur.next(D);
    w_.lm_head = (c.flags & kFlagTied) ? w_.token_embedding : cur.next(V * D);

    ctx_ = std::min(ctx_len, c.max_seq_len);
    s_.x.resize(D);
    s_.xb.resize(std::max(D, Q));
    s_.xb2.resize(D);
    s_.q.resize(Q);
    s_.hb.resize(H);
    s_.hb2.resize(H);
    s_.att.resize(size_t(c.n_heads) * ctx_);
    s_.logits.resize(V);
    s_.key_cache.resize(size_t(c.n_layers) * ctx_ * KV);
    s_.value_cache.resize(size_t(c.n_layers) * ctx_ * KV);
}

Model::~Model() {
    if (map_ && map_ != MAP_FAILED) munmap(map_, map_size_);
}

const float* Model::forward(int token, int pos, bool compute_logits) {
    const Config& c = cfg_;
    if (pos >= ctx_) throw std::runtime_error("context length exceeded");
    if (token < 0 || token >= c.vocab_size) throw std::runtime_error("token id out of range");
    const int D = c.dim, H = c.hidden_dim, Q = c.q_dim(), KV = c.kv_dim(), hd = c.head_dim;
    const int group = c.n_heads / c.n_kv_heads;  // query heads sharing one KV head
    const float scale = 1.0f / std::sqrt(float(hd));
    float* x = s_.x.data();
    float* xb = s_.xb.data();
    float* xb2 = s_.xb2.data();

    std::memcpy(x, w_.token_embedding + size_t(token) * D, D * sizeof(float));

    for (int l = 0; l < c.n_layers; ++l) {
        const LayerWeights& L = w_.layers[l];
        float* kc = s_.key_cache.data() + size_t(l) * ctx_ * KV;
        float* vc = s_.value_cache.data() + size_t(l) * ctx_ * KV;
        float* k = kc + size_t(pos) * KV;  // this token's key/value go straight into the cache
        float* v = vc + size_t(pos) * KV;

        // ---- attention ----
        ops::rmsnorm(xb, x, L.attn_norm, D, c.norm_eps);
        ops::matvec(s_.q.data(), L.wq, xb, Q, D);
        ops::matvec(k, L.wk, xb, KV, D);
        ops::matvec(v, L.wv, xb, KV, D);
        if (L.bq) {
            ops::add(s_.q.data(), L.bq, Q);
            ops::add(k, L.bk, KV);
            ops::add(v, L.bv, KV);
        }
        ops::rope(s_.q.data(), c.n_heads, hd, pos, c.rope_theta);
        ops::rope(k, c.n_kv_heads, hd, pos, c.rope_theta);

        for (int h = 0; h < c.n_heads; ++h) {
            const float* qh = s_.q.data() + h * hd;
            const int kvh = h / group;
            float* att = s_.att.data() + size_t(h) * ctx_;
            for (int t = 0; t <= pos; ++t) {
                const float* kt = kc + size_t(t) * KV + kvh * hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; ++i) dot += qh[i] * kt[i];
                att[t] = dot * scale;
            }
            ops::softmax(att, pos + 1);
            float* out = xb + h * hd;
            std::fill(out, out + hd, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                const float* vt = vc + size_t(t) * KV + kvh * hd;
                const float a = att[t];
                for (int i = 0; i < hd; ++i) out[i] += a * vt[i];
            }
        }
        ops::matvec(xb2, L.wo, xb, D, Q);
        ops::add(x, xb2, D);

        // ---- SwiGLU MLP ----
        ops::rmsnorm(xb, x, L.mlp_norm, D, c.norm_eps);
        ops::matvec(s_.hb.data(), L.w_gate, xb, H, D);
        ops::matvec(s_.hb2.data(), L.w_up, xb, H, D);
        ops::swiglu(s_.hb.data(), s_.hb2.data(), H);
        ops::matvec(xb2, L.w_down, s_.hb.data(), D, H);
        ops::add(x, xb2, D);
    }

    if (!compute_logits) return nullptr;
    ops::rmsnorm(x, x, w_.final_norm, D, c.norm_eps);
    ops::matvec(s_.logits.data(), w_.lm_head, x, c.vocab_size, D);
    return s_.logits.data();
}
