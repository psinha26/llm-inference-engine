// Unit tests: each op against NumPy reference vectors (tests/gen_op_vectors.py).
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ops/ops.h"

#ifndef OP_VECTORS_PATH
#error "OP_VECTORS_PATH must be defined by CMake"
#endif

namespace {

const std::map<std::string, std::vector<float>>& vectors() {
    static std::map<std::string, std::vector<float>> v = [] {
        std::map<std::string, std::vector<float>> m;
        std::ifstream f(OP_VECTORS_PATH, std::ios::binary);
        if (!f) throw std::runtime_error("missing " OP_VECTORS_PATH "; run tests/gen_op_vectors.py");
        char name[32];
        int32_t n;
        while (f.read(name, 32) && f.read(reinterpret_cast<char*>(&n), 4)) {
            std::vector<float> data(n);
            f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
            m[std::string(name, strnlen(name, 32))] = std::move(data);
        }
        return m;
    }();
    return v;
}

std::vector<float> get(const std::string& name) { return vectors().at(name); }

void expect_close(const std::vector<float>& got, const std::vector<float>& want, float tol) {
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i)
        EXPECT_NEAR(got[i], want[i], tol * std::max(1.0f, std::fabs(want[i]))) << "at index " << i;
}

}  // namespace

TEST(Ops, Matvec) {
    auto W = get("matvec.W"), x = get("matvec.x"), want = get("matvec.out");
    std::vector<float> out(want.size());
    ops::matvec(out.data(), W.data(), x.data(), int(out.size()), int(x.size()));
    expect_close(out, want, 1e-5f);
}

TEST(Ops, RMSNorm) {
    auto x = get("rmsnorm.x"), w = get("rmsnorm.w"), want = get("rmsnorm.out");
    std::vector<float> out(x.size());
    ops::rmsnorm(out.data(), x.data(), w.data(), int(x.size()), get("rmsnorm.eps")[0]);
    expect_close(out, want, 1e-5f);
}

TEST(Ops, RMSNormInPlace) {
    auto x = get("rmsnorm.x"), w = get("rmsnorm.w"), want = get("rmsnorm.out");
    ops::rmsnorm(x.data(), x.data(), w.data(), int(x.size()), get("rmsnorm.eps")[0]);
    expect_close(x, want, 1e-5f);
}

TEST(Ops, Rope) {
    auto x = get("rope.x"), p = get("rope.params"), want = get("rope.out");
    ops::rope(x.data(), int(p[0]), int(p[1]), int(p[2]), p[3]);
    expect_close(x, want, 1e-5f);
}

TEST(Ops, RopePositionZeroIsIdentity) {
    auto x = get("rope.x"), orig = x;
    ops::rope(x.data(), 3, 64, 0, 10000.0f);
    expect_close(x, orig, 0.0f);
}

TEST(Ops, SoftmaxStableForLargeLogits) {
    auto x = get("softmax.x"), want = get("softmax.out");
    ops::softmax(x.data(), int(x.size()));
    expect_close(x, want, 1e-5f);
    float sum = 0;
    for (float v : x) sum += v;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(Ops, SwiGLU) {
    auto a = get("swiglu.a"), b = get("swiglu.b"), want = get("swiglu.out");
    ops::swiglu(a.data(), b.data(), int(a.size()));
    expect_close(a, want, 1e-5f);
}

TEST(Ops, Add) {
    std::vector<float> a = {1, 2, 3}, b = {0.5f, -2, 10};
    ops::add(a.data(), b.data(), 3);
    expect_close(a, {1.5f, 0, 13}, 0.0f);
}
