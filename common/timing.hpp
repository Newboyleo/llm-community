#pragma once
// timing.hpp — GPU event timer + CPU wall clock.
//
// The GPU event timer is the workhorse of every "Performance Analysis" section
// in this lab. Two things matter:
//
//   1. cudaEventRecord on the SAME stream as the work you measure.
//   2. cudaEventSynchronize (or stream sync) before reading elapsed time.
//
// We never time CPU-side wall-clock for GPU work — that includes launch
// overhead and is meaningless. The CpuTimer below is only for setup or for
// code that is genuinely host-side.

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

namespace lab {

class GpuTimer {
   public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Record start onto a stream. Default stream is 0 (legacy default stream).
    void start(cudaStream_t stream = nullptr) { cudaEventRecord(start_, stream); }
    void stop(cudaStream_t stream = nullptr) { cudaEventRecord(stop_, stream); }

    // Block until the stop event has completed, then return elapsed ms.
    float elapsed_ms() {
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

   private:
    cudaEvent_t start_, stop_;
};

class CpuTimer {
   public:
    void start() { begin_ = std::chrono::steady_clock::now(); }
    double elapsed_ms() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - begin_).count();
    }

   private:
    std::chrono::steady_clock::time_point begin_;
};

// Pretty-print a bandwidth figure given bytes moved and elapsed milliseconds.
// Example: 1 GiB in 1.0 ms -> "8590.0 GB/s" (decimal GB, as nvidia-smi reports).
inline void print_bandwidth(const char* label, size_t bytes, float ms) {
    double gb = static_cast<double>(bytes) / 1e9;        // decimal GB
    double gbs = gb / (ms / 1000.0);
    double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    double gibs = gib / (ms / 1000.0);
    std::printf("[bw] %-24s %.3f ms  %.1f GB/s  (%.1f GiB/s)  %zu bytes\n",
                label, ms, gbs, gibs, bytes);
}

}  // namespace lab
