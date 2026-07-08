// lesson06-reduce-scatter/reduce_scatter.cu
//
// ReduceScatter: each rank r ends up owning chunk r of the element-wise sum.
// Implemented naive and ring. Ring is the first half of ring AllReduce.

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "checks.hpp"
#include "multigpu.hpp"
#include "print.hpp"
#include "timing.hpp"

__global__ void add_into(int* dst, const int* src, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) dst[i] += src[i];
}

static int initial_value(int rank, int chunk_id, int elem) {
    return rank * 1000000 + chunk_id * 1000 + (elem & 1023);
}

struct State {
    int n;
    int chunk;  // ints per chunk; L = n*chunk
    std::vector<int*> d;
    std::vector<int*> scratch;  // scratch[r] = one-chunk recv buffer on rank r
    std::vector<cudaStream_t> streams;
};

static State setup(int n, int chunk) {
    State s; s.n = n; s.chunk = chunk;
    int L = n * chunk;
    s.d.assign(n, nullptr);
    s.scratch.assign(n, nullptr);
    s.streams.assign(n, nullptr);
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMalloc(&s.d[r], L * sizeof(int)));
        LAB_CUDA(cudaMalloc(&s.scratch[r], chunk * sizeof(int)));
        LAB_CUDA(cudaStreamCreate(&s.streams[r]));
        std::vector<int> tmp(L);
        for (int c = 0; c < n; ++c) {
            for (int i = 0; i < chunk; ++i) {
                tmp[c * chunk + i] = initial_value(r, c, i);
            }
        }
        LAB_CUDA(cudaMemcpy(s.d[r], tmp.data(), L * sizeof(int), cudaMemcpyHostToDevice));
    }
    return s;
}
static void teardown(State& s) {
    for (int r = 0; r < s.n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaFree(s.d[r]));
        LAB_CUDA(cudaFree(s.scratch[r]));
        LAB_CUDA(cudaStreamDestroy(s.streams[r]));
    }
}

static bool verify(State& s) {
    for (int r = 0; r < s.n; ++r) {
        std::vector<int> host(s.chunk);
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMemcpy(host.data(), s.d[r] + r * s.chunk, s.chunk * sizeof(int),
                            cudaMemcpyDeviceToHost));
        for (int i = 0; i < s.chunk; ++i) {
            int expect = 0;
            for (int src = 0; src < s.n; ++src) {
                expect += initial_value(src, r, i);
            }
            if (host[i] != expect) return false;
        }
    }
    return true;
}

// naive: each src sends its chunk[dst] to dst; dst adds.
static float rs_naive(State& s) {
    int n = s.n, chunk = s.chunk;
    size_t cb = chunk * sizeof(int);
    LAB_CUDA(cudaSetDevice(0));
    lab::GpuTimer t;
    t.start(s.streams[0]);
    for (int dst = 0; dst < n; ++dst) {
        for (int src = 0; src < n; ++src) {
            if (src == dst) continue;
            LAB_CUDA(cudaSetDevice(dst));
            LAB_CUDA(cudaMemcpyPeerAsync(s.scratch[dst], dst,
                                         s.d[src] + dst * chunk, src,
                                         cb, s.streams[dst]));
            LAB_CUDA(cudaStreamSynchronize(s.streams[dst]));
            add_into<<<(chunk + 255) / 256, 256, 0, s.streams[dst]>>>(
                s.d[dst] + dst * chunk, s.scratch[dst], chunk);
            LAB_CUDA(cudaStreamSynchronize(s.streams[dst]));
        }
    }
    LAB_CUDA(cudaSetDevice(0));
    t.stop(s.streams[0]);
    return t.elapsed_ms();
}

// ring: n-1 steps. At step k, rank r forwards chunk (r-step-1) to r+1,
// receives chunk (r-step-2) from r-1 into scratch, adds into that slot.
static float rs_ring(State& s) {
    int n = s.n, chunk = s.chunk;
    size_t cb = chunk * sizeof(int);
    LAB_CUDA(cudaSetDevice(0));
    lab::GpuTimer t;
    t.start(s.streams[0]);
    for (int step = 0; step < n - 1; ++step) {
        for (int r = 0; r < n; ++r) {
            int next = (r + 1) % n;
            int send_chunk = (r - step - 1 + n) % n;
            // r sends its send_chunk slot to next's scratch
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaMemcpyPeerAsync(s.scratch[next], next,
                                         s.d[r] + send_chunk * chunk, r,
                                         cb, s.streams[r]));
        }
        for (int r = 0; r < n; ++r) {
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
        }
        // each receiver adds scratch into its recv_chunk slot
        for (int r = 0; r < n; ++r) {
            int recv_chunk = (r - step - 2 + n) % n;
            LAB_CUDA(cudaSetDevice(r));
            add_into<<<(chunk + 255) / 256, 256, 0, s.streams[r]>>>(
                s.d[r] + recv_chunk * chunk, s.scratch[r], chunk);
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
    int chunk = 1 << 16;  // 256 KiB ints per chunk
    if (argc > 1) chunk = std::atoi(argv[1]);
    LAB_CHECK(chunk > 0, "chunk must be positive");
    LAB_CHECK(chunk <= INT_MAX / n, "n*chunk overflows int");

    std::printf("==== lesson 06: reduce-scatter ====\n");
    lab::enable_all_peers(n);
    std::printf("n_gpus = %d\nL = %d ints (%.2f MiB per rank), chunk = %.2f MiB\n",
                n, n * chunk, n * chunk * sizeof(int) / (1024.0 * 1024.0),
                chunk * sizeof(int) / (1024.0 * 1024.0));

    auto run = [&](const char* name, auto fn) {
        State s = setup(n, chunk);
        // rs_naive/rs_ring time on streams[0], a device-0 stream.
        LAB_CUDA(cudaSetDevice(0));
        float ms = fn(s);
        bool ok = verify(s);
        std::printf("\n==== %s ====\nresult %s\n", name, ok ? "OK" : "MISMATCH");
        lab::print_bandwidth(name, (size_t)chunk * sizeof(int) * (n - 1) * n, ms);
        teardown(s);
    };

    run("naive reduce-scatter", [](State& s) { return rs_naive(s); });
    run("ring reduce-scatter",  [](State& s) { return rs_ring(s); });

    std::printf("\nlesson 06 done.\n");
    return 0;
}
