// lesson08-alltoall/alltoall.cu
//
// AllToAll: rank i sends block j to rank j. After the call, rank j holds
// block j from every rank i (a transpose of the block layout). This is the
// skeleton of MoE dispatch.

#include <cstdio>
#include <vector>

#include "checks.hpp"
#include "multigpu.hpp"
#include "print.hpp"
#include "timing.hpp"

struct State {
    int n, block;
    std::vector<int*> d;
    std::vector<cudaStream_t> streams;
};

static State setup(int n, int block) {
    State s; s.n = n; s.block = block;
    s.d.assign(n, nullptr);
    s.streams.assign(n, nullptr);
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMalloc(&s.d[r], n * block * sizeof(int)));
        LAB_CUDA(cudaStreamCreate(&s.streams[r]));
        // slot j on rank r = tag 10000*r + j  (the block r -> j)
        std::vector<int> tmp(n * block);
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < block; ++k)
                tmp[j * block + k] = 10000 * r + j;
        LAB_CUDA(cudaMemcpy(s.d[r], tmp.data(), tmp.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
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
static bool verify(State& s) {
    // after: rank j slot i == 10000*i + j
    for (int r = 0; r < s.n; ++r) {
        std::vector<int> host(s.n * s.block);
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMemcpy(host.data(), s.d[r], host.size() * sizeof(int),
                            cudaMemcpyDeviceToHost));
        for (int i = 0; i < s.n; ++i)
            for (int k = 0; k < s.block; ++k)
                if (host[i * s.block + k] != 10000 * i + r) return false;
    }
    return true;
}

static float a2a_naive(State& s) {
    int n = s.n, block = s.block;
    size_t bb = block * sizeof(int);
    LAB_CUDA(cudaSetDevice(0));
    lab::GpuTimer t;
    t.start(s.streams[0]);
    for (int src = 0; src < n; ++src) {
        for (int dst = 0; dst < n; ++dst) {
            if (src == dst) continue;  // self-block stays in place
            // rank src sends its slot[dst] to rank dst's slot[src]
            LAB_CUDA(cudaSetDevice(src));
            LAB_CUDA(cudaMemcpyPeerAsync(s.d[dst] + src * block, dst,
                                         s.d[src] + dst * block, src,
                                         bb, s.streams[src]));
        }
    }
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
    }
    LAB_CUDA(cudaSetDevice(0));
    t.stop(s.streams[0]);
    return t.elapsed_ms();
}

int main(int argc, char** argv) {
    int n = lab::require_gpus(2);
    if (n > 8) n = 8;
    int block = 1 << 16;  // 256 KiB ints per block
    if (argc > 1) block = std::atoi(argv[1]);

    std::printf("==== lesson 08: alltoall ====\n");
    lab::enable_all_peers(n);
    std::printf("n_gpus = %d\nblock = %d ints (%.2f KiB), total per rank = %.2f MiB\n",
                n, block, block * sizeof(int) / 1024.0,
                n * block * sizeof(int) / (1024.0 * 1024.0));

    State s = setup(n, block);
    // a2a_naive times on streams[0], a device-0 stream.
    LAB_CUDA(cudaSetDevice(0));
    // show "before" tags on rank 0
    {
        std::vector<int> host(n);
        for (int j = 0; j < n; ++j) host[j] = 10000 * 0 + j;
        lab::print_host("before r0 tags", host.data(), n, n);
    }

    float ms = a2a_naive(s);
    bool ok = verify(s);
    std::printf("\n==== naive alltoall ====\nresult %s\n", ok ? "OK" : "MISMATCH");
    {
        std::vector<int> host(n);
        LAB_CUDA(cudaSetDevice(0));
        LAB_CUDA(cudaMemcpy(host.data(), s.d[0], n * sizeof(int), cudaMemcpyDeviceToHost));
        lab::print_host("after  r0 tags", host.data(), n, n);
    }
    lab::print_bandwidth("alltoall", (size_t)n * block * sizeof(int) * (n - 1), ms);

    teardown(s);
    std::printf("\nlesson 08 done.\n");
    return 0;
}
