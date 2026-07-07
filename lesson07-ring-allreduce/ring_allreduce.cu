// lesson07-ring-allreduce/ring_allreduce.cu
//
// Ring AllReduce = ring ReduceScatter (lesson 6) + ring AllGather (lesson 4).
// Compared against a naive reduce-to-zero + broadcast.

#include <cstdio>
#include <vector>

#include "checks.hpp"
#include "multigpu.hpp"
#include "print.hpp"
#include "timing.hpp"

__global__ void add_into(int* dst, const int* src, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) dst[i] += src[i];
}

struct State {
    int n, chunk;  // L = n*chunk
    std::vector<int*> d;
    std::vector<int*> scratch;
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
    int expect = 0; for (int r = 0; r < s.n; ++r) expect += r;
    for (int r = 0; r < s.n; ++r) {
        std::vector<int> host(s.n * s.chunk);
        LAB_CUDA(cudaMemcpy(host.data(), s.d[r], host.size() * sizeof(int),
                            cudaMemcpyDeviceToHost));
        for (int v : host) if (v != expect) return false;
    }
    return true;
}

// ---- phase 1: ring ReduceScatter ----  (each rank ends owning chunk r = Σ)
static void ring_reduce_scatter(State& s) {
    int n = s.n, chunk = s.chunk;
    size_t cb = chunk * sizeof(int);
    for (int step = 0; step < n - 1; ++step) {
        for (int r = 0; r < n; ++r) {
            int next = (r + 1) % n;
            int send_chunk = (r - step + n) % n;
            LAB_CUDA(cudaMemcpyPeerAsync(s.scratch[next], next,
                                         s.d[r] + send_chunk * chunk, r,
                                         cb, s.streams[r]));
        }
        for (int r = 0; r < n; ++r) LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
        for (int r = 0; r < n; ++r) {
            int recv_chunk = (r - step - 1 + n) % n;
            add_into<<<(chunk + 255) / 256, 256, 0, s.streams[r]>>>(
                s.d[r] + recv_chunk * chunk, s.scratch[r], chunk);
        }
        for (int r = 0; r < n; ++r) LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
    }
}

// ---- phase 2: ring AllGather ----  (each rank forwards its owned Σ chunk)
static void ring_all_gather(State& s) {
    int n = s.n, chunk = s.chunk;
    size_t cb = chunk * sizeof(int);
    for (int step = 0; step < n - 1; ++step) {
        for (int r = 0; r < n; ++r) {
            int next = (r + 1) % n;
            int fwd_chunk = (r - step + n) % n;  // chunk I forward this step
            LAB_CUDA(cudaMemcpyPeerAsync(s.d[next] + fwd_chunk * chunk, next,
                                         s.d[r]     + fwd_chunk * chunk, r,
                                         cb, s.streams[r]));
        }
        for (int r = 0; r < n; ++r) LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
    }
}

static float ring_allreduce(State& s) {
    lab::GpuTimer t;
    t.start(s.streams[0]);
    ring_reduce_scatter(s);
    ring_all_gather(s);
    t.stop(s.streams[0]);
    return t.elapsed_ms();
}

// naive: reduce-to-0 (gather+add) then 0 broadcasts to all.
static float naive_allreduce(State& s) {
    int n = s.n, chunk = s.chunk;
    int L = n * chunk;
    size_t lb = L * sizeof(int);
    lab::GpuTimer t;
    t.start(s.streams[0]);
    // reduce to rank 0
    for (int r = 1; r < n; ++r) {
        LAB_CUDA(cudaMemcpyPeerAsync(s.scratch[0], 0, s.d[r], r, lb, s.streams[0]));
        LAB_CUDA(cudaStreamSynchronize(s.streams[0]));
        add_into<<<(L + 255) / 256, 256, 0, s.streams[0]>>>(s.d[0], s.scratch[0], L);
        LAB_CUDA(cudaStreamSynchronize(s.streams[0]));
    }
    // broadcast from rank 0
    for (int r = 1; r < n; ++r) {
        LAB_CUDA(cudaMemcpyPeerAsync(s.d[r], r, s.d[0], 0, lb, s.streams[0]));
    }
    LAB_CUDA(cudaStreamSynchronize(s.streams[0]));
    t.stop(s.streams[0]);
    return t.elapsed_ms();
}

int main(int argc, char** argv) {
    int n = lab::require_gpus(2);
    if (n > 8) n = 8;
    int chunk = 1 << 16;  // 256 KiB ints
    if (argc > 1) chunk = std::atoi(argv[1]);

    std::printf("==== lesson 07: ring allreduce ====\n");
    lab::enable_all_peers(n);
    std::printf("n_gpus = %d\nL = %d ints (%.2f MiB per rank)\n",
                n, n * chunk, n * chunk * sizeof(int) / (1024.0 * 1024.0));
    int expect = 0; for (int r = 0; r < n; ++r) expect += r;
    std::printf("expected (every index) = %d\n", expect);

    auto run = [&](const char* name, auto fn) {
        State s = setup(n, chunk);
        // GpuTimer events are created on the *current* device's context, and
        // ring_allreduce/naive_allreduce time on streams[0] (a device-0 stream).
        // setup() ends with cudaSetDevice(n-1), so without this the timer's
        // events are created on device n-1 but recorded on device 0's stream —
        // cudaEventRecord silently fails (GpuTimer doesn't LAB_CUDA-wrap it),
        // elapsed_ms() returns 0.0f, and bandwidth prints as inf.
        LAB_CUDA(cudaSetDevice(0));
        float ms = fn(s);
        bool ok = verify(s);
        std::printf("\n==== %s ====\nresult %s\n", name, ok ? "OK" : "MISMATCH");
        size_t moved = (size_t)n * chunk * sizeof(int) * (n - 1) * 2;  // rough
        lab::print_bandwidth(name, moved, ms);
        teardown(s);
    };

    run("naive allreduce", [](State& s) { return naive_allreduce(s); });
    run("ring allreduce",  [](State& s) { return ring_allreduce(s); });

    std::printf("\nlesson 07 done.\n");
    return 0;
}
