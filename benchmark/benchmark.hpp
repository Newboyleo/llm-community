#pragma once
// benchmark.hpp — a reusable "sweep over sizes, report bandwidth" loop.
//
// Almost every lesson's Performance Analysis section wants the same thing:
//   for each size in a geometric list:
//       warmup
//       time N iterations
//       report min latency + peak bandwidth
//
// Rather than copy this loop into 20 lessons, we factor it here. The lesson
// supplies a callable that performs ONE timed iteration on the default stream.

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

#include "timing.hpp"

namespace lab {

// A single size + the bytes moved per iteration (may differ from the size for
// collectives where each rank contributes/collects a slice).
struct BenchPoint {
    size_t size_bytes;     // size of the buffer involved
    size_t moved_bytes;    // bytes actually crossing the interconnect
};

// Default size sweep: 1 KiB -> 1 GiB, doubling.
inline std::vector<size_t> default_sizes() {
    std::vector<size_t> v;
    for (size_t s = 1024; s <= 1ull << 30; s <<= 1) v.push_back(s);
    return v;
}

// Run `iters` iterations of `fn` (after `warmup` warmup iters) and report.
// `fn` must do its work on `stream` and not synchronize the device itself.
inline void sweep(const char* title,
                  const std::vector<size_t>& sizes,
                  size_t moved_bytes_per_size,  // bytes moved per iter for THIS op
                  const std::function<void(cudaStream_t)>& fn,
                  int warmup = 5,
                  int iters = 20) {
    std::printf("\n==== %s ====\n", title);
    std::printf("%-12s %10s %12s %12s\n", "size", "min_ms", "GB/s", "GiB/s");
    std::printf("------------ ---------- ------------ ------------\n");

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    GpuTimer t;

    for (size_t s : sizes) {
        (void)s;  // size is fixed by `fn` setup; s is just for the table label
        for (int i = 0; i < warmup; ++i) fn(stream);
        cudaStreamSynchronize(stream);

        t.start(stream);
        for (int i = 0; i < iters; ++i) fn(stream);
        t.stop(stream);
        float ms = t.elapsed_ms() / iters;

        double gbs = (moved_bytes_per_size / 1e9) / (ms / 1000.0);
        double gibs = (moved_bytes_per_size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
        std::printf("%-12zu %10.4f %12.1f %12.1f\n", s, ms, gbs, gibs);
    }
    cudaStreamDestroy(stream);
}

}  // namespace lab
