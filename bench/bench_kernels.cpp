// Kernel microbenchmark: effective weight bandwidth (GB/s) of each matvec kernel when the weights
// are cache-resident ("hot") versus streamed from DRAM ("cold", a working set far larger than the
// caches, like a real forward pass), plus the fixed cost of one thread-pool parallel region.
//
//   ./bench_kernels [threads]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ops/kernels.h"
#include "ops/thread_pool.h"

using kernels::Impl;

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Mats {
    int rows, cols, count;
    std::vector<float> f;
    std::vector<int8_t> q;
    std::vector<float> d;
    std::vector<float> ft;   // tiled copies (see ops/kernels.h for the layout)
    std::vector<int8_t> qt;
    std::vector<float> dt;
};

Mats make(int rows, int cols, int count) {
    Mats m{rows, cols, count, {}, {}, {}, {}, {}, {}};
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 0.02f);
    m.f.resize(size_t(rows) * cols * count);
    for (auto& v : m.f) v = nd(rng);
    m.q.resize(m.f.size());
    m.d.resize(m.f.size() / kernels::kQ8Block);
    kernels::quantize_q8(m.f.data(), int(m.f.size()), m.q.data(), m.d.data());
    // Tile each matrix (rows is a multiple of 4 here, so no padding is needed).
    const int B = kernels::kQ8Block, C = kernels::kTileColsF32, nb = cols / B;
    m.ft.resize(m.f.size());
    m.qt.resize(m.q.size());
    m.dt.resize(m.d.size());
    for (int k = 0; k < count; ++k) {
        const size_t base = size_t(k) * rows * cols, dbase = base / B;
        for (int r = 0; r < rows; ++r) {
            const int g = r / 4, j = r % 4;
            for (int c = 0; c < cols; ++c) {
                m.ft[base + size_t(g) * 4 * cols + (size_t(c / C) * 4 + j) * C + c % C] = m.f[base + size_t(r) * cols + c];
                m.qt[base + size_t(g) * 4 * cols + (size_t(c / B) * 4 + j) * B + c % B] = m.q[base + size_t(r) * cols + c];
            }
            for (int b = 0; b < nb; ++b) m.dt[dbase + size_t(g) * 4 * nb + b * 4 + j] = m.d[dbase + size_t(r) * nb + b];
        }
    }
    return m;
}

// Runs `iters` matvecs cycling through the `count` matrices; returns weight GB/s.
double run(ThreadPool& pool, const Mats& m, Impl impl, bool q8, int iters, bool tiled = false) {
    std::vector<float> x(m.cols, 0.5f), out(m.rows);
    std::vector<int8_t> xq(m.cols);
    std::vector<float> xd(m.cols / kernels::kQ8Block);
    kernels::quantize_q8(x.data(), m.cols, xq.data(), xd.data());
    const size_t per = size_t(m.rows) * m.cols;
    auto one = [&](int i) {
        const int k = i % m.count;
        if (q8 && tiled) {
            const int8_t* W = m.qt.data() + k * per;
            const float* D = m.dt.data() + k * per / kernels::kQ8Block;
            pool.parallel_for(m.rows, [&](int r0, int r1) {
                kernels::matvec_q8_tiled(impl, out.data(), W, D, xq.data(), xd.data(), r0, r1, m.rows, m.cols);
            }, 4);
        } else if (tiled) {
            const float* W = m.ft.data() + k * per;
            pool.parallel_for(m.rows, [&](int r0, int r1) {
                kernels::matvec_f32_tiled(impl, out.data(), W, x.data(), r0, r1, m.rows, m.cols);
            }, 4);
        } else if (q8) {
            const int8_t* W = m.q.data() + k * per;
            const float* D = m.d.data() + k * per / kernels::kQ8Block;
            pool.parallel_for(m.rows, [&](int r0, int r1) {
                kernels::matvec_q8(impl, out.data(), W, D, xq.data(), xd.data(), r0, r1, m.cols);
            }, 4);
        } else {
            const float* W = m.f.data() + k * per;
            pool.parallel_for(m.rows, [&](int r0, int r1) {
                kernels::matvec_f32(impl, out.data(), W, x.data(), r0, r1, m.cols);
            }, 4);
        }
    };
    for (int i = 0; i < m.count; ++i) one(i);  // warm-up pass
    double t0 = now_s();
    for (int i = 0; i < iters; ++i) one(i);
    double dt = now_s() - t0;
    const double bytes = q8 ? per * (1.0 + 4.0 / kernels::kQ8Block) : per * 4.0;
    return bytes * iters / dt / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 1;
    ThreadPool pool(threads);
    std::printf("SIMD backend: %s, threads: %d\n\n", kernels::simd_name(), threads);

    // MLP-shaped matrices (SmolLM2-360M up_proj: 2560 x 960).
    Mats hot = make(2560, 960, 1);    // 9.8 MB fp32: mostly cache-resident on Apple Silicon
    Mats cold = make(2560, 960, 48);  // 470 MB fp32: every pass streams from DRAM
    std::printf("%-16s %-6s %12s %12s\n", "kernel", "format", "hot GB/s", "cold GB/s");
    struct Row { const char* name; Impl impl; bool tiled; };
    const Row variants[] = {{"naive", Impl::Naive, false}, {"simd", Impl::Simd, false},
                            {"blocked", Impl::Blocked, false}, {"blocked+tiled", Impl::Blocked, true}};
    for (const Row& v : variants) {
        for (bool q8 : {false, true}) {
            int iters = v.impl == Impl::Naive ? 20 : 200;
            std::printf("%-16s %-6s %12.1f %12.1f\n", v.name, q8 ? "int8" : "fp32",
                        run(pool, hot, v.impl, q8, iters, v.tiled), run(pool, cold, v.impl, q8, iters, v.tiled));
        }
    }

    // Fixed cost of a parallel region with no work: what every matmul pays for threading.
    const int n = 20000;
    double t0 = now_s();
    for (int i = 0; i < n; ++i) pool.parallel_for(threads, [](int, int) {});
    std::printf("\nempty parallel_for: %.2f us per region\n", (now_s() - t0) / n * 1e6);
    return 0;
}
