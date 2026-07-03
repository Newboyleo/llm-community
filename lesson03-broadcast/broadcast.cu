// lesson03-broadcast/broadcast.cu
//
// Broadcast rank 0's buffer to all ranks, three ways: naive, ring, tree.
// Uses only cudaMemcpyPeerAsync (lesson 2's primitive). The lesson is that the
// *schedule* of peer copies determines latency, not the copies themselves.

#include <cmath>
#include <cstdio>
#include <vector>

#include "checks.hpp"
#include "multigpu.hpp"
#include "print.hpp"
#include "timing.hpp"

// Allocate one buffer per device, fill rank 0 with 1,2,3,...
static std::vector<int*> alloc_all(int n, size_t bytes) {
    std::vector<int*> d(n, nullptr);
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMalloc(&d[r], bytes));
        LAB_CUDA(cudaMemset(d[r], 0, bytes));
    }
    // Fill rank 0 with a recognizable pattern.
    std::vector<int> tmp(bytes / sizeof(int));
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = static_cast<int>(i + 1);
    LAB_CUDA(cudaMemcpy(d[0], tmp.data(), bytes, cudaMemcpyHostToDevice));
    return d;
}

static void free_all(std::vector<int*>& d) {
    for (size_t r = 0; r < d.size(); ++r) {
        if (d[r]) {
            LAB_CUDA(cudaSetDevice(static_cast<int>(r)));
            LAB_CUDA(cudaFree(d[r]));
        }
    }
}

// Verify every rank's buffer equals rank 0's.
static bool verify_all_match(const std::vector<int*>& d, size_t bytes) {
    int n = static_cast<int>(d.size());
    std::vector<int> ref(bytes / sizeof(int));
    LAB_CUDA(cudaMemcpy(ref.data(), d[0], bytes, cudaMemcpyDeviceToHost));
    for (int r = 1; r < n; ++r) {
        std::vector<int> got(bytes / sizeof(int));
        LAB_CUDA(cudaMemcpy(got.data(), d[r], bytes, cudaMemcpyDeviceToHost));
        if (got != ref) return false;
    }
    return true;
}

// --- naive: rank 0 sends to each rank serially on one stream ----------------
static float bcast_naive(std::vector<int*>& d, int n, size_t bytes, cudaStream_t s) {
    lab::GpuTimer t;
    t.start(s);
    for (int dst = 1; dst < n; ++dst) {
        LAB_CUDA(cudaMemcpyPeerAsync(d[dst], dst, d[0], 0, bytes, s));
    }
    LAB_CUDA(cudaStreamSynchronize(s));
    t.stop(s);
    return t.elapsed_ms();
}

// --- ring: 0 -> 1 -> 2 -> ... -> n-1, each hop on its own stream -------------
static float bcast_ring(std::vector<int*>& d, int n, size_t bytes,
                        std::vector<cudaStream_t>& streams) {
    lab::GpuTimer t;
    t.start(streams[0]);
    for (int r = 0; r + 1 < n; ++r) {
        LAB_CUDA(cudaMemcpyPeerAsync(d[r + 1], r + 1, d[r], r, bytes, streams[r]));
    }
    for (int r = 0; r < n; ++r) LAB_CUDA(cudaStreamSynchronize(streams[r]));
    t.stop(streams[0]);
    return t.elapsed_ms();
}

// --- tree: doubling. step s: rank r sends to r | (1<<s) ---------------------
static float bcast_tree(std::vector<int*>& d, int n, size_t bytes,
                        std::vector<cudaStream_t>& streams) {
    lab::GpuTimer t;
    t.start(streams[0]);
    for (int step = 0; (1 << step) < n; ++step) {
        for (int r = 0; r < n; ++r) {
            int partner = r | (1 << step);
            if (r < partner && partner < n) {
                LAB_CUDA(cudaMemcpyPeerAsync(d[partner], partner, d[r], r, bytes,
                                             streams[r]));
            }
        }
        for (int r = 0; r < n; ++r) LAB_CUDA(cudaStreamSynchronize(streams[r]));
    }
    t.stop(streams[0]);
    return t.elapsed_ms();
}

int main(int argc, char** argv) {
    int n = lab::require_gpus(2);
    size_t bytes = 1ull << 22;  // 4 MiB of ints
    if (argc > 1) bytes = static_cast<size_t>(std::atoll(argv[1])) * sizeof(int);
    if (n > 8) n = 8;  // keep the demo bounded

    std::printf("==== lesson 03: broadcast ====\n");
    lab::enable_all_peers(n);
    std::printf("n_gpus = %d\nbytes = %.2f MiB\n", n, bytes / (1024.0 * 1024.0));

    // one stream per rank so ring/tree hops can overlap
    std::vector<cudaStream_t> streams(n);
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaStreamCreate(&streams[r]));
    }

    auto d = alloc_all(n, bytes);
    lab::print_device("src", d[0], bytes / sizeof(int), 4);

    auto run = [&](const char* name, auto fn) {
        // reset receivers
        for (int r = 1; r < n; ++r) {
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaMemset(d[r], 0, bytes));
        }
        float ms = fn(d, n, bytes, streams);
        bool ok = verify_all_match(d, bytes);
        std::printf("\n==== %s ====\nall ranks match src: %s\n", name, ok ? "YES" : "NO");
        lab::print_bandwidth(name, bytes * (n - 1), ms);
    };

    run("naive broadcast", [&](auto& d, int n, size_t b, auto& s) {
        return bcast_naive(d, n, b, s[0]);
    });
    run("ring broadcast",  bcast_ring);
    run("tree broadcast",  bcast_tree);

    free_all(d);
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaStreamDestroy(streams[r]));
    }
    std::printf("\nlesson 03 done.\n");
    return 0;
}
