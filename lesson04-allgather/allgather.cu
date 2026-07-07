// lesson04-allgather/allgather.cu
//
// AllGather: each rank r owns slice r; after the call every rank owns the
// concatenation [s0, s1, ..., s_{n-1}]. Implemented naive and ring.

#include <cstdio>
#include <vector>

#include "checks.hpp"
#include "multigpu.hpp"
#include "print.hpp"
#include "timing.hpp"

// Layout: d[r] is an int* of n*slice_ints. Slot r is pre-filled with rank r's
// data; other slots start at 0. After AllGather, every slot of every rank is
// filled: slot k on rank r == slice k.
struct State {
    int n;
    size_t slice_ints;
    std::vector<int*> d;       // d[r] -> device buffer on rank r
    std::vector<cudaStream_t> streams;
};

static State setup(int n, size_t slice_ints) {
    State s;
    s.n = n;
    s.slice_ints = slice_ints;
    s.d.assign(n, nullptr);
    s.streams.assign(n, nullptr);
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMalloc(&s.d[r], n * slice_ints * sizeof(int)));
        LAB_CUDA(cudaMemset(s.d[r], 0, n * slice_ints * sizeof(int)));
        LAB_CUDA(cudaStreamCreate(&s.streams[r]));
        // fill own slice with a recognizable value: 1000*r + index
        std::vector<int> tmp(slice_ints);
        for (size_t i = 0; i < slice_ints; ++i) tmp[i] = 1000 * r + static_cast<int>(i);
        LAB_CUDA(cudaMemcpy(s.d[r] + r * slice_ints, tmp.data(),
                            slice_ints * sizeof(int), cudaMemcpyHostToDevice));
    }
    return s;
}

static void teardown(State& s) {
    for (int r = 0; r < s.n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaFree(s.d[r]));
        LAB_CUDA(cudaStreamDestroy(s.streams[r]));
    }
}

static bool verify(const State& s) {
    // rank 0's buffer should be [s0, s1, ..., s_{n-1}] = [0,1,.. ; 1000,1001,.. ; ...]
    std::vector<int> host(s.n * s.slice_ints);
    LAB_CUDA(cudaSetDevice(0));
    LAB_CUDA(cudaMemcpy(host.data(), s.d[0], host.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
    for (int k = 0; k < s.n; ++k) {
        for (size_t i = 0; i < s.slice_ints; ++i) {
            int expect = 1000 * k + static_cast<int>(i);
            if (host[k * s.slice_ints + i] != expect) return false;
        }
    }
    return true;
}

// naive: each src sends its own slice to every other dst. [s0, s1, ..., s_{n-1}]
static float ag_naive(State& s) {
    size_t sb = s.slice_ints * sizeof(int);
    LAB_CUDA(cudaSetDevice(0));
    lab::GpuTimer t;
    t.start(s.streams[0]);
    for (int src = 0; src < s.n; ++src) {
        for (int dst = 0; dst < s.n; ++dst) {
            if (src == dst) continue;
            LAB_CUDA(cudaMemcpyPeerAsync(s.d[dst] + src * s.slice_ints, dst,
                                         s.d[src] + src * s.slice_ints, src,
                                         sb, s.streams[0]));
        }
    }
    LAB_CUDA(cudaStreamSynchronize(s.streams[0]));
    t.stop(s.streams[0]);
    return t.elapsed_ms();
}

// ring: n-1 steps. At step k, rank r forwards the slice it received last step.
// We use a per-rank stream + barrier between steps.
static float ag_ring(State& s) {
    int n = s.n;
    size_t sb = s.slice_ints * sizeof(int);
    LAB_CUDA(cudaSetDevice(0));
    lab::GpuTimer t;
    t.start(s.streams[0]);
    for (int step = 0; step < n - 1; ++step) {
        for (int r = 0; r < n; ++r) {
            int next = (r + 1) % n;
            // slice I'm forwarding this step = my own slice, shifted by step+1
            int fwd_slice = (r - step + n) % n;
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaMemcpyPeerAsync(s.d[next] + fwd_slice * s.slice_ints, next,
                                         s.d[r]     + fwd_slice * s.slice_ints, r,
                                         sb, s.streams[r]));
        }
        for (int r = 0; r < n; ++r) {
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
        }
    }
    LAB_CUDA(cudaSetDevice(0));
    t.stop(s.streams[0]);
    return t.elapsed_ms();
}

int main(int argc, char** argv) {
    int n = lab::require_gpus(2);
    if (n > 8) n = 8;
    size_t slice_ints = 1 << 16;  // 256 KiB of ints per slice
    if (argc > 1) slice_ints = static_cast<size_t>(std::atoll(argv[1]));

    std::printf("==== lesson 04: allgather ====\n");
    lab::enable_all_peers(n);
    std::printf("n_gpus = %d\nslice = %zu ints (%.2f KiB), total = %.2f MiB\n",
                n, slice_ints, slice_ints * sizeof(int) / 1024.0,
                n * slice_ints * sizeof(int) / (1024.0 * 1024.0));

    auto run = [&](const char* name, auto fn) {
        State s = setup(n, slice_ints);
        // ag_naive/ag_ring time on streams[0], a device-0 stream.
        LAB_CUDA(cudaSetDevice(0));
        float ms = fn(s);
        bool ok = verify(s);
        std::printf("\n==== %s ====\nresult %s\n", name, ok ? "OK" : "MISMATCH");
        lab::print_bandwidth(name, n * slice_ints * sizeof(int) * (n - 1), ms);
        teardown(s);
    };

    run("naive allgather", [](State& s) { return ag_naive(s); });
    run("ring allgather",  [](State& s) { return ag_ring(s); });

    std::printf("\nlesson 04 done.\n");
    return 0;
}
