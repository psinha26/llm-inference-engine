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
    TensorCursor(const char* base, size_t size, bool q8, bool tiled)
        : base_(base), size_(size), off_(kHeaderBytes), q8_(q8), tiled_(tiled) {}

    const float* f32(size_t n) { return static_cast<const float*>(take(n * sizeof(float))); }

    Mat mat(int rows, int cols) {
        Mat m;
        m.rows = rows;
        m.cols = cols;
        m.tiled = tiled_;
        // Tiled matrices are stored with rows padded up to a multiple of 4.
        const int stored_rows = tiled_ ? (rows + kernels::kTileRows - 1) / kernels::kTileRows * kernels::kTileRows : rows;
        if (tiled_ && cols % kernels::kTileColsF32) throw std::runtime_error("tiled weights need cols divisible by 8");
        const size_t n = size_t(stored_rows) * cols;
        if (q8_) {
            if (cols % kernels::kQ8Block) throw std::runtime_error("int8 weights need cols divisible by 32");
            m.q = static_cast<const int8_t*>(take(n));
            m.d = f32(n / kernels::kQ8Block);
        } else {
            m.f = f32(n);
        }
        return m;
    }

private:
    const void* take(size_t bytes) {
        off_ = (off_ + kTensorAlign - 1) / kTensorAlign * kTensorAlign;
        if (off_ + bytes > size_) throw std::runtime_error("model file is truncated");
        const void* p = base_ + off_;
        off_ += bytes;
        return p;
    }
    const char* base_;
    size_t size_, off_;
    bool q8_, tiled_;
};

}  // namespace

void Mat::row(int r, float* out) const {
    const int B = kernels::kQ8Block, T = kernels::kTileRows;
    if (!tiled) {
        if (q) {
            const int8_t* qr = q + size_t(r) * cols;
            const float* dr = d + size_t(r) * (cols / B);
            for (int i = 0; i < cols; ++i) out[i] = dr[i / B] * qr[i];
        } else {
            std::memcpy(out, f + size_t(r) * cols, cols * sizeof(float));
        }
        return;
    }
    const int g = r / T, j = r % T;
    if (q) {  // [group][block][4 rows x 32], scales [group][block][4]
        const int8_t* qg = q + size_t(g) * T * cols;
        const float* dg = d + size_t(g) * T * (cols / B);
        for (int b = 0; b < cols / B; ++b)
            for (int i = 0; i < B; ++i) out[b * B + i] = dg[b * T + j] * qg[(size_t(b) * T + j) * B + i];
    } else {  // [group][8-col chunk][4 rows x 8]
        const int C = kernels::kTileColsF32;
        const float* fg = f + size_t(g) * T * cols;
        for (int c = 0; c < cols / C; ++c)
            for (int i = 0; i < C; ++i) out[c * C + i] = fg[(size_t(c) * T + j) * C + i];
    }
}

Model::Model(const std::string& path, int ctx_len, const ModelOptions& opts) : opts_(opts) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open model file " + path);
    struct stat st;
    fstat(fd, &st);
    map_size_ = size_t(st.st_size);
    if (opts.no_mmap) {
        // One big allocation, aligned to 2 MB so the kernel can back it with huge pages.
        const size_t huge = size_t(2) << 20;
        const size_t alloc = (map_size_ + huge - 1) / huge * huge;
        map_ = std::aligned_alloc(huge, alloc);
        if (!map_) throw std::runtime_error("out of memory loading " + path);
        owned_ = true;
#ifdef MADV_HUGEPAGE
        madvise(map_, alloc, MADV_HUGEPAGE);
#endif
        size_t done = 0;
        while (done < map_size_) {
            ssize_t n = read(fd, static_cast<char*>(map_) + done, map_size_ - done);
            if (n <= 0) throw std::runtime_error("read failed for " + path);
            done += size_t(n);
        }
        close(fd);
    } else {
        map_ = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (map_ == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
    }
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
    const int D = c.dim, H = c.hidden_dim, Q = c.q_dim(), KV = c.kv_dim(), V = c.vocab_size;
    TensorCursor cur(base, map_size_, c.q8(), c.tiled());
    w_.token_embedding = cur.mat(V, D);
    w_.layers.resize(c.n_layers);
    for (auto& L : w_.layers) {
        L.attn_norm = cur.f32(D);
        L.wq = cur.mat(Q, D);
        L.wk = cur.mat(KV, D);
        L.wv = cur.mat(KV, D);
        if (c.flags & kFlagQkvBias) {
            L.bq = cur.f32(Q);
            L.bk = cur.f32(KV);
            L.bv = cur.f32(KV);
        }
        L.wo = cur.mat(D, Q);
        L.mlp_norm = cur.f32(D);
        L.w_gate = cur.mat(H, D);
        L.w_up = cur.mat(H, D);
        L.w_down = cur.mat(D, H);
    }
    w_.final_norm = cur.f32(D);
    w_.lm_head = (c.flags & kFlagTied) ? w_.token_embedding : cur.mat(V, D);

    ctx_ = std::min(ctx_len, c.max_seq_len);
    const int widest = std::max({D, Q, H});
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
    s_.xq.resize(widest);
    s_.xd.resize(widest / kernels::kQ8Block + 1);

    pool_ = std::make_unique<ThreadPool>(opts_.threads);
}

Model::~Model() {
    if (owned_) std::free(map_);
    else if (map_ && map_ != MAP_FAILED) munmap(map_, map_size_);
}

// Matrix-vector products that share the same input x, e.g. the Q/K/V projections or the MLP
// gate/up projections. All of them run in ONE parallel region: their rows are laid end to end
// (each padded to a multiple of 4) and split evenly across threads. Fewer, larger regions mean
// less synchronisation, and the int8 path quantizes x only once.
void Model::matmul(std::initializer_list<MatOut> jobs, const float* x) {
    const kernels::Impl impl = opts_.impl;
    const int T = kernels::kTileRows;
    const int cols = jobs.begin()->W->cols;
    int total = 0;
    bool any_q8 = false;
    for (const MatOut& j : jobs) {
        total += (j.W->rows + T - 1) / T * T;
        any_q8 = any_q8 || j.W->q;
    }
    if (any_q8) kernels::quantize_q8(x, cols, s_.xq.data(), s_.xd.data());
    const int8_t* xq = s_.xq.data();
    const float* xd = s_.xd.data();

    pool_->parallel_for(total, [&](int r0, int r1) {
        int start = 0;
        for (const MatOut& j : jobs) {
            const Mat& W = *j.W;
            const int span = (W.rows + T - 1) / T * T;
            const int a = std::max(r0, start) - start, b = std::min(r1, start + span) - start;
            start += span;
            if (a >= b) continue;
            const int lo = a, hi = std::min(b, W.rows);  // [lo, hi) local rows; hi may cut padding
            if (lo >= hi) continue;
            if (W.q && W.tiled) kernels::matvec_q8_tiled(impl, j.out, W.q, W.d, xq, xd, lo, hi, W.rows, cols);
            else if (W.q) kernels::matvec_q8(impl, j.out, W.q, W.d, xq, xd, lo, hi, cols);
            else if (W.tiled) kernels::matvec_f32_tiled(impl, j.out, W.f, x, lo, hi, W.rows, cols);
            else kernels::matvec_f32(impl, j.out, W.f, x, lo, hi, cols);
        }
    }, T);
}

// Grouped-query attention for the token at `pos`, one head per task.
void Model::attention(int layer, int pos) {
    const Config& c = cfg_;
    const int KV = c.kv_dim(), hd = c.head_dim;
    const int group = c.n_heads / c.n_kv_heads;  // query heads sharing one KV head
    const float scale = 1.0f / std::sqrt(float(hd));
    const float* kc = s_.key_cache.data() + size_t(layer) * ctx_ * KV;
    const float* vc = s_.value_cache.data() + size_t(layer) * ctx_ * KV;

    pool_->parallel_for(c.n_heads, [&](int h0, int h1) {
        for (int h = h0; h < h1; ++h) {
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
            float* out = s_.xb.data() + h * hd;
            std::fill(out, out + hd, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                const float* vt = vc + size_t(t) * KV + kvh * hd;
                const float a = att[t];
                for (int i = 0; i < hd; ++i) out[i] += a * vt[i];
            }
        }
    });
}

const float* Model::forward(int token, int pos, bool compute_logits) {
    const Config& c = cfg_;
    if (pos >= ctx_) throw std::runtime_error("context length exceeded");
    if (token < 0 || token >= c.vocab_size) throw std::runtime_error("token id out of range");
    const int D = c.dim, H = c.hidden_dim, Q = c.q_dim(), KV = c.kv_dim(), hd = c.head_dim;
    float* x = s_.x.data();
    float* xb = s_.xb.data();
    float* xb2 = s_.xb2.data();

    w_.token_embedding.row(token, x);  // embedding lookup

    for (int l = 0; l < c.n_layers; ++l) {
        const LayerWeights& L = w_.layers[l];
        float* k = s_.key_cache.data() + (size_t(l) * ctx_ + pos) * KV;  // written straight into the cache
        float* v = s_.value_cache.data() + (size_t(l) * ctx_ + pos) * KV;

        // ---- attention ----
        ops::rmsnorm(xb, x, L.attn_norm, D, c.norm_eps);
        matmul({{s_.q.data(), &L.wq}, {k, &L.wk}, {v, &L.wv}}, xb);
        if (L.bq) {
            ops::add(s_.q.data(), L.bq, Q);
            ops::add(k, L.bk, KV);
            ops::add(v, L.bv, KV);
        }
        ops::rope(s_.q.data(), c.n_heads, hd, pos, c.rope_theta);
        ops::rope(k, c.n_kv_heads, hd, pos, c.rope_theta);
        attention(l, pos);
        matmul({{xb2, &L.wo}}, xb);
        ops::add(x, xb2, D);

        // ---- SwiGLU MLP ----
        ops::rmsnorm(xb, x, L.mlp_norm, D, c.norm_eps);
        matmul({{s_.hb.data(), &L.w_gate}, {s_.hb2.data(), &L.w_up}}, xb);
        ops::swiglu(s_.hb.data(), s_.hb2.data(), H);
        matmul({{xb2, &L.w_down}}, s_.hb.data());
        ops::add(x, xb2, D);
    }

    if (!compute_logits) return nullptr;
    ops::rmsnorm(x, x, w_.final_norm, D, c.norm_eps);
    matmul({{s_.logits.data(), &w_.lm_head}}, x);
    return s_.logits.data();
}
