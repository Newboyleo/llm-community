// lesson05-reduce/reduce.cu
//
// Reduce (sum) across n GPUs: rank 0 ends with x0+x1+...+x_{n-1}.
// Implemented naive (gather-then-sum) and tree.

#include <cstdio>
#include <vector>

#include "checks.hpp"
#include "multigpu.hpp"
#include "print.hpp"
#include "timing.hpp"

// dst[i] += src[i]
__global__ void add_into(int* dst, const int* src, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) dst[i] += src[i];
}

struct State {
    int n;
    int L;
    std::vector<int*> d;        // d[r] = rank r's buffer
    std::vector<int*> scratch;  // scratch[r] = rank r's recv scratch
    std::vector<cudaStream_t> streams;
};

static State setup(int n, int L) {
    State s; s.n = n; s.L = L;
    s.d.assign(n, nullptr);
    s.scratch.assign(n, nullptr);
    s.streams.assign(n, nullptr);
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMalloc(&s.d[r], L * sizeof(int)));
        LAB_CUDA(cudaMalloc(&s.scratch[r], L * sizeof(int)));
        LAB_CUDA(cudaStreamCreate(&s.streams[r]));
        // fill rank r with constant value r -> sum = 0+1+...+(n-1)
        std::vector<int> tmp(L, r);
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
    int expect = 0;
    for (int r = 0; r < s.n; ++r) expect += r;
    std::vector<int> host(s.L);
    LAB_CUDA(cudaMemcpy(host.data(), s.d[0], s.L * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < s.L; ++i) if (host[i] != expect) return false;
    return true;
}

// naive: copy each r>0 into rank0's scratch, add_into rank0's buffer.
static float reduce_naive(State& s) {
    size_t bytes = s.L * sizeof(int);
    lab::GpuTimer t;
    t.start(s.streams[0]);
    for (int r = 1; r < s.n; ++r) {
        LAB_CUDA(cudaMemcpyPeerAsync(s.scratch[0], 0, s.d[r], r, bytes, s.streams[0]));
        LAB_CUDA(cudaStreamSynchronize(s.streams[0]));
        add_into<<<(s.L + 255) / 256, 256, 0, s.streams[0]>>>(s.d[0], s.scratch[0], s.L);
        LAB_CUDA(cudaStreamSynchronize(s.streams[0]));
    }
    t.stop(s.streams[0]);
    return t.elapsed_ms();
}

// tree: step s, rank r pairs with r ^ (1<<s). Higher sends, lower adds.
static float reduce_tree(State& s) {
    size_t bytes = s.L * sizeof(int);
    int n = s.n;
    lab::GpuTimer t;
    t.start(s.streams[0]);
    for (int step = 0; (1 << step) < n; ++step) {
        for (int r = 0; r < n; ++r) {
            int partner = r ^ (1 << step);
            if (partner >= n) continue;
            if (r > partner) {  // sender
                LAB_CUDA(cudaMemcpyPeerAsync(s.scratch[partner], partner, s.d[r], r,
                                             bytes, s.streams[r]));
            }
        }
        for (int r = 0; r < n; ++r) LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
        for (int r = 0; r < n; ++r) {
            int partner = r ^ (1 << step);
            if (partner >= n || r > partner) continue;  // receiver r < partner
            add_into<<<(s.L + 255) / 256, 256, 0, s.streams[r]>>>(s.d[r], s.scratch[r], s.L);
        }
        for (int r = 0; r < n; ++r) LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
    }
    t.stop(s.streams[0]);
    return t.elapsed_ms();
}

int main(int argc, char** argv) {
    int n = lab::require_gpus(2);
    if (n > 8) n = 8;
    int L = 1 << 18;  // 1 MiB of ints
    if (argc > 1) L = std::atoi(argv[1]);

    std::printf("==== lesson 05: reduce ====\n");
    lab::enable_all_peers(n);
    std::printf("n_gpus = %d\nL = %d ints (%.2f MiB per rank)\n",
                n, L, L * sizeof(int) / (1024.0 * 1024.0));
    int expect = 0; for (int r = 0; r < n; ++r) expect += r;
    std::printf("expected sum (every index) = %d\n", expect);

    auto run = [&](const char* name, auto fn) {
        State s = setup(n, L);
        float ms = fn(s);
        bool ok = verify(s);
        std::printf("\n==== %s ====\nresult %s\n", name, ok ? "OK" : "MISMATCH");
        lab::print_bandwidth(name, (size_t)L * sizeof(int) * (n - 1), ms);
        teardown(s);
    };

    run("naive gather-then-sum", [](State& s) { return reduce_naive(s); });
    run("tree reduce",           [](State& s) { return reduce_tree(s); });

    std::printf("\nlesson 05 done.\n");
    return 0;
}
