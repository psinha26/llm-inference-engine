#include "ops/thread_pool.h"

#include <algorithm>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
static inline void cpu_relax() { _mm_pause(); }
#elif defined(__aarch64__)
static inline void cpu_relax() { asm volatile("yield" ::: "memory"); }
#else
static inline void cpu_relax() {}
#endif

namespace {

// How long an idle worker spins before sleeping. This must outlast the serial work between two
// parallel regions (norms, RoPE, residual adds), or every region pays a full futex wake-up.
// It is time-based because a spin *count* means very different things on different CPUs: the
// first version spun 20,000 iterations, which is only a few microseconds on Apple Silicon, so
// workers fell asleep between regions and 4 threads ran slower than 1.
constexpr auto kSpinTime = std::chrono::microseconds(500);

// Spin until pred() is true or kSpinTime passes; returns pred().
template <class Pred>
bool spin_until(Pred pred) {
    const auto deadline = std::chrono::steady_clock::now() + kSpinTime;
    for (;;) {
        for (int i = 0; i < 64; ++i) {  // check the clock only every 64 polls
            if (pred()) return true;
            cpu_relax();
        }
        if (std::chrono::steady_clock::now() > deadline) return pred();
    }
}

}  // namespace

ThreadPool::ThreadPool(int n_threads) : n_threads_(std::max(1, n_threads)) {
    for (int i = 1; i < n_threads_; ++i) workers_.emplace_back(&ThreadPool::worker_loop, this, i);
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        generation_.fetch_add(1, std::memory_order_release);
    }
    cv_start_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::run_chunk(int id) {
    // Split [0, n) into n_threads_ chunks whose boundaries are multiples of align_.
    const int units = (n_ + align_ - 1) / align_;
    const int per = units / n_threads_, extra = units % n_threads_;
    const int u0 = id * per + std::min(id, extra);
    const int u1 = u0 + per + (id < extra ? 1 : 0);
    const int begin = std::min(n_, u0 * align_), end = std::min(n_, u1 * align_);
    if (begin < end) (*fn_)(begin, end);
}

void ThreadPool::worker_loop(int id) {
    uint64_t seen = 0;
    for (;;) {
        // Spin first: the next parallel region usually arrives within microseconds.
        spin_until([&] { return generation_.load(std::memory_order_acquire) != seen; });
        uint64_t gen = generation_.load(std::memory_order_acquire);
        if (gen == seen) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_start_.wait(lk, [&] { return generation_.load(std::memory_order_acquire) != seen; });
            gen = generation_.load(std::memory_order_acquire);
        }
        seen = gen;
        if (stop_) return;

        run_chunk(id);

        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(mu_);  // pairs with the caller's predicate check
            cv_done_.notify_one();
        }
    }
}

void ThreadPool::parallel_for(int n, const std::function<void(int, int)>& fn, int align) {
    if (n <= 0) return;
    if (n_threads_ == 1) {
        fn(0, n);
        return;
    }
    fn_ = &fn;
    n_ = n;
    align_ = std::max(1, align);
    remaining_.store(n_threads_ - 1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(mu_);
        generation_.fetch_add(1, std::memory_order_release);
    }
    cv_start_.notify_all();

    run_chunk(0);  // the caller does its share instead of idling

    if (!spin_until([&] { return remaining_.load(std::memory_order_acquire) == 0; })) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
    }
}
