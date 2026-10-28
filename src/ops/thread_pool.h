// Fixed-size thread pool built on std::thread + condition variables (no OpenMP).
//
// parallel_for splits [0, n) into one contiguous chunk per thread and runs them in parallel;
// the calling thread does chunk 0 itself, so a pool of size N uses N-1 worker threads.
//
// A token's forward pass issues ~250 small parallel regions, so wake-up latency matters.
// Workers therefore spin (for up to 500 us) on an atomic generation counter before falling back
// to a condition-variable sleep: back-to-back regions are picked up in well under a microsecond,
// while an idle pool (e.g. between requests) costs no CPU.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return n_threads_; }

    // fn(begin, end) is called once per thread with disjoint ranges covering [0, n).
    // Chunk boundaries are multiples of `align` (useful for kernels that process rows in groups).
    void parallel_for(int n, const std::function<void(int, int)>& fn, int align = 1);

private:
    void worker_loop(int id);
    void run_chunk(int id);

    int n_threads_;
    std::vector<std::thread> workers_;

    std::mutex mu_;
    std::condition_variable cv_start_, cv_done_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> remaining_{0};
    bool stop_ = false;

    // Current job (written by the caller before bumping generation_).
    const std::function<void(int, int)>* fn_ = nullptr;
    int n_ = 0, align_ = 1;
};
