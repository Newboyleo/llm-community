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
    std::vector<int*> scratch;     // ring: one-chunk recv buffer
    std::vector<int*> hd_scratch;  // halving-doubling: up to (n/2)*chunk recv buffer
    int* naive_scratch;            // naive: full-buffer recv scratch on rank 0
    std::vector<cudaStream_t> streams;
};

static State setup(int n, int chunk) {
    State s; s.n = n; s.chunk = chunk;
    int L = n * chunk;
    s.d.assign(n, nullptr);
    s.scratch.assign(n, nullptr);
    s.hd_scratch.assign(n, nullptr);
    s.naive_scratch = nullptr;
    s.streams.assign(n, nullptr);
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMalloc(&s.d[r], L * sizeof(int)));
        LAB_CUDA(cudaMalloc(&s.scratch[r], chunk * sizeof(int)));
        // halving-doubling's largest single exchange is n/2 chunks (the first
        // reduce step and the last allgather step), so that's all the recv
        // buffer it needs. We only run it for power-of-two n (see main()).
        LAB_CUDA(cudaMalloc(&s.hd_scratch[r], (size_t)(n / 2) * chunk * sizeof(int)));
        LAB_CUDA(cudaStreamCreate(&s.streams[r]));
        std::vector<int> tmp(L, r);
        LAB_CUDA(cudaMemcpy(s.d[r], tmp.data(), L * sizeof(int), cudaMemcpyHostToDevice));
    }
    LAB_CUDA(cudaSetDevice(0));
    LAB_CUDA(cudaMalloc(&s.naive_scratch, L * sizeof(int)));
    return s;
}
static void teardown(State& s) {
    LAB_CUDA(cudaSetDevice(0));
    LAB_CUDA(cudaFree(s.naive_scratch));
    for (int r = 0; r < s.n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaFree(s.d[r]));
        LAB_CUDA(cudaFree(s.scratch[r]));
        LAB_CUDA(cudaFree(s.hd_scratch[r]));
        LAB_CUDA(cudaStreamDestroy(s.streams[r]));
    }
}
static bool verify(State& s) {
    int expect = 0; for (int r = 0; r < s.n; ++r) expect += r;
    for (int r = 0; r < s.n; ++r) {
        std::vector<int> host(s.n * s.chunk);
        LAB_CUDA(cudaSetDevice(r));
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
            int send_chunk = (r - step - 1 + n) % n;
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaMemcpyPeerAsync(s.scratch[next], next,
                                         s.d[r] + send_chunk * chunk, r,
                                         cb, s.streams[r]));
        }
        for (int r = 0; r < n; ++r) {
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
        }
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
}

// ---- phase 2: ring AllGather ----  (each rank forwards its owned Σ chunk)
static void ring_all_gather(State& s) {
    int n = s.n, chunk = s.chunk;
    size_t cb = chunk * sizeof(int);
    for (int step = 0; step < n - 1; ++step) {
        for (int r = 0; r < n; ++r) {
            int next = (r + 1) % n;
            int fwd_chunk = (r - step + n) % n;  // chunk I forward this step
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaMemcpyPeerAsync(s.d[next] + fwd_chunk * chunk, next,
                                         s.d[r]     + fwd_chunk * chunk, r,
                                         cb, s.streams[r]));
        }
        for (int r = 0; r < n; ++r) {
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
        }
    }
}

static float ring_allreduce(State& s) {
    LAB_CUDA(cudaSetDevice(0));
    lab::GpuTimer t;
    t.start(s.streams[0]);
    ring_reduce_scatter(s);
    ring_all_gather(s);
    LAB_CUDA(cudaSetDevice(0));
    t.stop(s.streams[0]);
    return t.elapsed_ms();
}

// ---- halving-doubling AllReduce ----  (no ring wraparound; O(log n) hops)
//
// Recursive halving (ReduceScatter) + recursive doubling (AllGather). Each rank
// tracks a window [off, off+len) of chunks it currently "owns"; len starts at n
// and halves (reduce) then grows back from 1 to n (allgather).
//
// Reduce step (len -> len/2): pair rank r with partner = r XOR (len/2). The two
// share the same window [off, off+len). Each sends the half of the window closer
// to its partner and reduces what it receives into the half it keeps:
//   r < partner : keep lower [off, off+len/2), send upper [off+len/2, off+len)
//   r > partner : keep upper, send lower. Window shrinks to the kept half.
// After log2(n) steps every rank owns one chunk = the sum of that chunk's slice
// across all ranks (rank r ends owning chunk r).
//
// AllGather step (len -> 2*len): pair r with partner = r XOR len. Each owns len
// consecutive chunks starting at off[r]; they exchange their owned runs and each
// writes the partner's run into the adjacent half, doubling its owned region.
//
// Total: 2·log2(n) steps; per-rank send = S·(n-1)/n — same bandwidth-optimal
// volume as the ring, but O(log n) latency instead of O(n).
// Requires n to be a power of two (its XOR-partner scheme only tiles cleanly for
// n = 2^k; NCCL pads non-power-of-two counts in production).

static float halving_doubling_allreduce(State& s) {
    int n = s.n, chunk = s.chunk;
    std::vector<int> off(n, 0);  // current window base (in chunks) per rank
    LAB_CUDA(cudaSetDevice(0));
    lab::GpuTimer t;
    t.start(s.streams[0]);

    // ---- reduce half: recursive halving (len: n -> n/2 -> ... -> 1) ----
    for (int len = n; len > 1; len >>= 1) {
        int half = len >> 1;                       // chunks each side sends/keeps
        size_t hb = (size_t)half * chunk * sizeof(int);
        for (int r = 0; r < n; ++r) {
            int partner = r ^ half;                // same window, opposite sub-half
            int send_off = (r < partner) ? (off[r] + half) : off[r];
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaMemcpyPeerAsync(s.hd_scratch[partner], partner,
                                         s.d[r] + send_off * chunk, r,
                                         hb, s.streams[r]));
        }
        for (int r = 0; r < n; ++r) {
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
        }
        for (int r = 0; r < n; ++r) {
            int partner = r ^ half;
            int keep_off = (r < partner) ? off[r] : (off[r] + half);
            LAB_CUDA(cudaSetDevice(r));
            add_into<<<(half * chunk + 255) / 256, 256, 0, s.streams[r]>>>(
                s.d[r] + keep_off * chunk, s.hd_scratch[r], half * chunk);
            off[r] = keep_off;                     // window shrinks to the kept half
        }
        for (int r = 0; r < n; ++r) {
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
        }
    }

    // ---- allgather half: recursive doubling (len: 1 -> 2 -> ... -> n) ----
    for (int len = 1; len < n; len <<= 1) {
        size_t lb = (size_t)len * chunk * sizeof(int);
        for (int r = 0; r < n; ++r) {
            int partner = r ^ len;                 // adjacent len-block, same 2*len-block
            // send our owned run [off[r], off[r]+len); partner writes it next to theirs
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaMemcpyPeerAsync(s.hd_scratch[partner], partner,
                                         s.d[r] + off[r] * chunk, r,
                                         lb, s.streams[r]));
        }
        for (int r = 0; r < n; ++r) {
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaStreamSynchronize(s.streams[r]));
        }
        for (int r = 0; r < n; ++r) {
            int partner = r ^ len;
            // the half we lack is the partner's run, adjacent to ours in the merged block
            int recv_off = (r < partner) ? (off[r] + len) : (off[r] - len);
            LAB_CUDA(cudaSetDevice(r));
            LAB_CUDA(cudaMemcpyAsync(s.d[r] + recv_off * chunk, s.hd_scratch[r],
                                     lb, cudaMemcpyDeviceToDevice, s.streams[r]));
            if (r > partner) off[r] -= len;        // window base moves down to merged base
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

// naive: reduce-to-0 (gather+add) then 0 broadcasts to all.
static float naive_allreduce(State& s) {
    int n = s.n, chunk = s.chunk;
    int L = n * chunk;
    size_t lb = L * sizeof(int);
    LAB_CUDA(cudaSetDevice(0));
    lab::GpuTimer t;
    t.start(s.streams[0]);
    // reduce to rank 0
    for (int r = 1; r < n; ++r) {
        LAB_CUDA(cudaMemcpyPeerAsync(s.naive_scratch, 0, s.d[r], r, lb, s.streams[0]));
        LAB_CUDA(cudaStreamSynchronize(s.streams[0]));
        add_into<<<(L + 255) / 256, 256, 0, s.streams[0]>>>(s.d[0], s.naive_scratch, L);
        LAB_CUDA(cudaStreamSynchronize(s.streams[0]));
    }
    // broadcast from rank 0
    for (int r = 1; r < n; ++r) {
        LAB_CUDA(cudaMemcpyPeerAsync(s.d[r], r, s.d[0], 0, lb, s.streams[0]));
    }
    LAB_CUDA(cudaStreamSynchronize(s.streams[0]));
    LAB_CUDA(cudaSetDevice(0));
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
        // All timing uses streams[0], a device-0 stream.
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

    // halving-doubling requires n to be a power of two (its partner = r XOR d
    // scheme only tiles cleanly when n = 2^k). Pad up to the next power of two
    // would require ghost ranks; instead we just skip it for odd n and note why.
    bool pow2 = (n & (n - 1)) == 0;
    if (pow2) {
        run("halving-doubling allreduce",
            [](State& s) { return halving_doubling_allreduce(s); });
    } else {
        std::printf("\n(halving-doubling skipped: n=%d is not a power of two)\n", n);
    }

    std::printf("\nlesson 07 done.\n");
    return 0;
}
