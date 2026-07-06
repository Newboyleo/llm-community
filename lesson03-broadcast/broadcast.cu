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

// --- ring: 0 -> 1 -> 2 -> ... -> n-1 ----------------------------------------
// Each hop r (d[r] -> d[r+1]) reads what hop r-1 wrote, so the hops carry a
// real data dependency and must be ordered. We put each hop on its own stream
// and chain them with events: hop r waits for hop r-1's "ready" event before
// reading d[r]. (For a single un-chunked buffer this is no faster than a
// single stream — the ordering is mandatory, not a parallelism win. The win
// comes in lesson 7 when the payload is split into pipelined chunks.)
static float bcast_ring(std::vector<int*>& d, int n, size_t bytes,
                        std::vector<cudaStream_t>& streams) {
    lab::GpuTimer t;
    t.start(streams[0]);
    std::vector<cudaEvent_t> ready(n);
    for (int r = 0; r < n; ++r)
        LAB_CUDA(cudaEventCreateWithFlags(&ready[r], cudaEventDisableTiming));
    for (int r = 0; r + 1 < n; ++r) {
        if (r > 0) LAB_CUDA(cudaStreamWaitEvent(streams[r], ready[r - 1], 0));
        LAB_CUDA(cudaMemcpyPeerAsync(d[r + 1], r + 1, d[r], r, bytes, streams[r]));
        LAB_CUDA(cudaEventRecord(ready[r], streams[r]));
    }
    for (int r = 0; r < n; ++r) LAB_CUDA(cudaStreamSynchronize(streams[r]));
    t.stop(streams[0]);
    for (int r = 0; r < n; ++r) LAB_CUDA(cudaEventDestroy(ready[r]));
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
        // GpuTimer events are created on the *current* device's context, so make
        // it device 0 — every fn below times on streams[0] (a device-0 stream).
        // Without this, elapsed_ms() silently returns 0.0f (create/record device
        // mismatch) and bandwidth prints as inf.
        LAB_CUDA(cudaSetDevice(0));
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
