// lesson08-alltoall/alltoall.cu
//
// A more general AllToAllv demo.
//
// Instead of treating AllToAll as "transpose an n x n matrix of equal blocks",
// this file models the MoE shape directly:
//
//   1. Context: counts/offsets/streams/buffers for every GPU pair.
//   2. dispatch: local tokens are bucketed by destination GPU, then AllToAllv.
//   3. combine: expert results travel back with the transposed plan, then
//      each source GPU scatters them back to token order.
//
// It is still intentionally small: host-orchestrated cudaMemcpyPeerAsync calls,
// no NCCL/NVSHMEM, and top-1 routing only. Lessons 16-17 add routing kernels
// and low-latency device-initiated transport.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "checks.hpp"
#include "multigpu.hpp"
#include "timing.hpp"

struct A2AContext {
    int n = 0;
    int tokens_per_rank = 0;
    int hidden_dim = 0;
    int total_tokens = 0;

    // Flattened as [src * n + dst].
    std::vector<int> counts;
    std::vector<int> dispatch_send_offsets;
    std::vector<int> dispatch_recv_offsets;
    std::vector<int> combine_counts;
    std::vector<int> combine_send_offsets;
    std::vector<int> combine_recv_offsets;

    // Per-token routing metadata, flattened as [rank * tokens_per_rank + t].
    std::vector<int> h_route;
    std::vector<int> h_slot_in_bucket;

    std::vector<cudaStream_t> streams;
    std::vector<float*> d_tokens;
    std::vector<float*> d_sendbuf;
    std::vector<float*> d_recvbuf;
    std::vector<float*> d_returnbuf;
    std::vector<float*> d_output;
    std::vector<int*> d_route;
    std::vector<int*> d_slot_in_bucket;
    std::vector<int*> d_dispatch_send_base;
};

static int route_token(int src_rank, int local_token, int n) {
    // Deterministic but intentionally uneven enough to make counts[src][dst]
    // vary. Replace this with top-k gating output in a real MoE layer.
    return (src_rank + ((local_token * 17 + local_token / 7 + 3) % n)) % n;
}

static int row_sum(const std::vector<int>& m, int n, int row) {
    int total = 0;
    for (int dst = 0; dst < n; ++dst) total += m[row * n + dst];
    return total;
}

static int col_sum(const std::vector<int>& m, int n, int col) {
    int total = 0;
    for (int src = 0; src < n; ++src) total += m[src * n + col];
    return total;
}

static size_t remote_bytes(const std::vector<int>& counts, int n, int hidden_dim) {
    size_t items = 0;
    for (int src = 0; src < n; ++src) {
        for (int dst = 0; dst < n; ++dst) {
            if (src != dst) items += (size_t)counts[src * n + dst];
        }
    }
    return items * hidden_dim * sizeof(float);
}

static void print_count_matrix(const char* title, const std::vector<int>& counts, int n) {
    std::printf("\n%s\n       ", title);
    for (int dst = 0; dst < n; ++dst) std::printf("dst%-5d", dst);
    std::printf("total\n");
    for (int src = 0; src < n; ++src) {
        std::printf("src%-3d ", src);
        int total = 0;
        for (int dst = 0; dst < n; ++dst) {
            int c = counts[src * n + dst];
            total += c;
            std::printf("%-8d", c);
        }
        std::printf("%d\n", total);
    }
}

__global__ void gather_dispatch_kernel(const float* __restrict__ tokens,
                                       const int* __restrict__ route,
                                       const int* __restrict__ slot_in_bucket,
                                       const int* __restrict__ send_base,
                                       float* __restrict__ sendbuf,
                                       int tokens_per_rank, int hidden_dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens_per_rank) return;

    int dst = route[t];
    int slot = slot_in_bucket[t];
    int packed_token = send_base[dst] + slot;
    for (int d = 0; d < hidden_dim; ++d) {
        sendbuf[(size_t)packed_token * hidden_dim + d] =
            tokens[(size_t)t * hidden_dim + d];
    }
}

__global__ void expert_compute_kernel(float* __restrict__ recvbuf,
                                      int tokens, int hidden_dim,
                                      int expert_rank) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) return;

    // Element 0 carries the global token id used by combine scatter. Keep it.
    // The remaining hidden values get a visible expert-rank transform.
    for (int d = 1; d < hidden_dim; ++d) {
        recvbuf[(size_t)t * hidden_dim + d] += 1000.0f * expert_rank;
    }
}

__global__ void scatter_combine_kernel(const float* __restrict__ returnbuf,
                                       float* __restrict__ output,
                                       int return_tokens,
                                       int tokens_per_rank,
                                       int hidden_dim) {
    int packed = blockIdx.x * blockDim.x + threadIdx.x;
    if (packed >= return_tokens) return;

    int global_token = (int)returnbuf[(size_t)packed * hidden_dim];
    int local_token = global_token % tokens_per_rank;
    for (int d = 0; d < hidden_dim; ++d) {
        output[(size_t)local_token * hidden_dim + d] =
            returnbuf[(size_t)packed * hidden_dim + d];
    }
}

static void build_plan(A2AContext& c) {
    int n = c.n;
    int T = c.tokens_per_rank;

    c.counts.assign(n * n, 0);
    c.h_route.assign(n * T, 0);
    c.h_slot_in_bucket.assign(n * T, 0);

    for (int src = 0; src < n; ++src) {
        std::vector<int> cursor(n, 0);
        for (int t = 0; t < T; ++t) {
            int dst = route_token(src, t, n);
            c.h_route[src * T + t] = dst;
            c.h_slot_in_bucket[src * T + t] = cursor[dst]++;
            c.counts[src * n + dst]++;
        }
    }

    c.dispatch_send_offsets.assign(n * n, 0);
    c.dispatch_recv_offsets.assign(n * n, 0);
    for (int src = 0; src < n; ++src) {
        int acc = 0;
        for (int dst = 0; dst < n; ++dst) {
            c.dispatch_send_offsets[src * n + dst] = acc;
            acc += c.counts[src * n + dst];
        }
    }
    for (int dst = 0; dst < n; ++dst) {
        int acc = 0;
        for (int src = 0; src < n; ++src) {
            c.dispatch_recv_offsets[src * n + dst] = acc;
            acc += c.counts[src * n + dst];
        }
    }

    // Combine is dispatch with src/dst transposed. The send offsets point into
    // each expert GPU's dispatch recvbuf; recv offsets build the source GPU's
    // return buffer, grouped by expert GPU.
    c.combine_counts.assign(n * n, 0);
    c.combine_send_offsets.assign(n * n, 0);
    c.combine_recv_offsets.assign(n * n, 0);
    for (int expert = 0; expert < n; ++expert) {
        for (int origin = 0; origin < n; ++origin) {
            c.combine_counts[expert * n + origin] = c.counts[origin * n + expert];
            c.combine_send_offsets[expert * n + origin] =
                c.dispatch_recv_offsets[origin * n + expert];
        }
    }
    for (int origin = 0; origin < n; ++origin) {
        int acc = 0;
        for (int expert = 0; expert < n; ++expert) {
            c.combine_recv_offsets[expert * n + origin] = acc;
            acc += c.counts[origin * n + expert];
        }
    }
}

static A2AContext setup(int n, int tokens_per_rank, int hidden_dim) {
    A2AContext c;
    c.n = n;
    c.tokens_per_rank = tokens_per_rank;
    c.hidden_dim = hidden_dim;
    c.total_tokens = n * tokens_per_rank;
    build_plan(c);

    c.streams.assign(n, nullptr);
    c.d_tokens.assign(n, nullptr);
    c.d_sendbuf.assign(n, nullptr);
    c.d_recvbuf.assign(n, nullptr);
    c.d_returnbuf.assign(n, nullptr);
    c.d_output.assign(n, nullptr);
    c.d_route.assign(n, nullptr);
    c.d_slot_in_bucket.assign(n, nullptr);
    c.d_dispatch_send_base.assign(n, nullptr);

    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaStreamCreate(&c.streams[r]));

        size_t local_bytes = (size_t)tokens_per_rank * hidden_dim * sizeof(float);
        size_t global_bytes = (size_t)c.total_tokens * hidden_dim * sizeof(float);
        LAB_CUDA(cudaMalloc(&c.d_tokens[r], local_bytes));
        LAB_CUDA(cudaMalloc(&c.d_sendbuf[r], local_bytes));
        LAB_CUDA(cudaMalloc(&c.d_recvbuf[r], global_bytes));
        LAB_CUDA(cudaMalloc(&c.d_returnbuf[r], local_bytes));
        LAB_CUDA(cudaMalloc(&c.d_output[r], local_bytes));
        LAB_CUDA(cudaMalloc(&c.d_route[r], tokens_per_rank * sizeof(int)));
        LAB_CUDA(cudaMalloc(&c.d_slot_in_bucket[r], tokens_per_rank * sizeof(int)));
        LAB_CUDA(cudaMalloc(&c.d_dispatch_send_base[r], n * sizeof(int)));

        std::vector<float> h_tokens((size_t)tokens_per_rank * hidden_dim);
        for (int t = 0; t < tokens_per_rank; ++t) {
            int global_token = r * tokens_per_rank + t;
            h_tokens[(size_t)t * hidden_dim] = (float)global_token;
            for (int d = 1; d < hidden_dim; ++d) {
                h_tokens[(size_t)t * hidden_dim + d] = (float)(global_token * 10 + d);
            }
        }

        LAB_CUDA(cudaMemcpy(c.d_tokens[r], h_tokens.data(), local_bytes,
                            cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(c.d_route[r], &c.h_route[r * tokens_per_rank],
                            tokens_per_rank * sizeof(int), cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(c.d_slot_in_bucket[r], &c.h_slot_in_bucket[r * tokens_per_rank],
                            tokens_per_rank * sizeof(int), cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(c.d_dispatch_send_base[r], &c.dispatch_send_offsets[r * n],
                            n * sizeof(int), cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemset(c.d_sendbuf[r], 0, local_bytes));
        LAB_CUDA(cudaMemset(c.d_recvbuf[r], 0, global_bytes));
        LAB_CUDA(cudaMemset(c.d_returnbuf[r], 0, local_bytes));
        LAB_CUDA(cudaMemset(c.d_output[r], 0, local_bytes));
    }
    return c;
}

static void teardown(A2AContext& c) {
    for (int r = 0; r < c.n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaFree(c.d_tokens[r]));
        LAB_CUDA(cudaFree(c.d_sendbuf[r]));
        LAB_CUDA(cudaFree(c.d_recvbuf[r]));
        LAB_CUDA(cudaFree(c.d_returnbuf[r]));
        LAB_CUDA(cudaFree(c.d_output[r]));
        LAB_CUDA(cudaFree(c.d_route[r]));
        LAB_CUDA(cudaFree(c.d_slot_in_bucket[r]));
        LAB_CUDA(cudaFree(c.d_dispatch_send_base[r]));
        LAB_CUDA(cudaStreamDestroy(c.streams[r]));
    }
}

static float alltoallv(A2AContext& c,
                       const std::vector<float*>& srcbufs,
                       const std::vector<float*>& dstbufs,
                       const std::vector<int>& counts,
                       const std::vector<int>& send_offsets,
                       const std::vector<int>& recv_offsets) {
    int n = c.n;
    int D = c.hidden_dim;

    LAB_CUDA(cudaSetDevice(0));
    lab::GpuTimer timer;
    timer.start(c.streams[0]);

    for (int src = 0; src < n; ++src) {
        for (int dst = 0; dst < n; ++dst) {
            int cnt = counts[src * n + dst];
            if (cnt == 0) continue;

            const float* send_ptr =
                srcbufs[src] + (size_t)send_offsets[src * n + dst] * D;
            float* recv_ptr =
                dstbufs[dst] + (size_t)recv_offsets[src * n + dst] * D;
            size_t bytes = (size_t)cnt * D * sizeof(float);

            LAB_CUDA(cudaSetDevice(src));
            if (src == dst) {
                LAB_CUDA(cudaMemcpyAsync(recv_ptr, send_ptr, bytes,
                                         cudaMemcpyDeviceToDevice,
                                         c.streams[src]));
            } else {
                LAB_CUDA(cudaMemcpyPeerAsync(recv_ptr, dst, send_ptr, src,
                                             bytes, c.streams[src]));
            }
        }
    }

    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaStreamSynchronize(c.streams[r]));
    }

    LAB_CUDA(cudaSetDevice(0));
    timer.stop(c.streams[0]);
    return timer.elapsed_ms();
}

static float dispatch(A2AContext& c) {
    int T = c.tokens_per_rank;
    int D = c.hidden_dim;
    int threads = 256;

    for (int r = 0; r < c.n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        gather_dispatch_kernel<<<(T + threads - 1) / threads, threads>>>(
            c.d_tokens[r], c.d_route[r], c.d_slot_in_bucket[r],
            c.d_dispatch_send_base[r], c.d_sendbuf[r], T, D);
        LAB_CUDA_SYNC();
    }

    return alltoallv(c, c.d_sendbuf, c.d_recvbuf, c.counts,
                     c.dispatch_send_offsets, c.dispatch_recv_offsets);
}

static void run_experts(A2AContext& c) {
    int threads = 256;
    for (int expert_rank = 0; expert_rank < c.n; ++expert_rank) {
        int tokens = col_sum(c.counts, c.n, expert_rank);
        if (tokens > 0) {
            LAB_CUDA(cudaSetDevice(expert_rank));
            expert_compute_kernel<<<(tokens + threads - 1) / threads, threads>>>(
                c.d_recvbuf[expert_rank], tokens, c.hidden_dim, expert_rank);
            LAB_CUDA_SYNC();
        }
        std::printf("GPU%d expert step processed %d tokens\n", expert_rank, tokens);
    }
}

static float combine(A2AContext& c) {
    float ms = alltoallv(c, c.d_recvbuf, c.d_returnbuf, c.combine_counts,
                         c.combine_send_offsets, c.combine_recv_offsets);

    int threads = 256;
    for (int r = 0; r < c.n; ++r) {
        int return_tokens = row_sum(c.counts, c.n, r);
        LAB_CUDA(cudaSetDevice(r));
        scatter_combine_kernel<<<(return_tokens + threads - 1) / threads, threads>>>(
            c.d_returnbuf[r], c.d_output[r], return_tokens,
            c.tokens_per_rank, c.hidden_dim);
        LAB_CUDA_SYNC();
    }
    return ms;
}

static bool verify(const A2AContext& c) {
    bool ok = true;
    int T = c.tokens_per_rank;
    int D = c.hidden_dim;

    for (int r = 0; r < c.n; ++r) {
        std::vector<float> out((size_t)T * D);
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMemcpy(out.data(), c.d_output[r],
                            out.size() * sizeof(float), cudaMemcpyDeviceToHost));

        for (int t = 0; t < T; ++t) {
            int global_token = r * T + t;
            int expert_rank = c.h_route[r * T + t];
            float got_id = out[(size_t)t * D];
            if ((int)got_id != global_token) {
                ok = false;
                break;
            }
            for (int d = 1; d < D; ++d) {
                float expected = (float)(global_token * 10 + d) + 1000.0f * expert_rank;
                float got = out[(size_t)t * D + d];
                if (std::fabs(got - expected) > 1e-3f) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
    }
    return ok;
}

static void print_rank0_sample(const A2AContext& c) {
    int show = c.tokens_per_rank < 8 ? c.tokens_per_rank : 8;
    std::vector<float> out((size_t)show * c.hidden_dim);
    LAB_CUDA(cudaSetDevice(0));
    LAB_CUDA(cudaMemcpy(out.data(), c.d_output[0],
                        out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("\nrank0 output sample:\n");
    for (int t = 0; t < show; ++t) {
        int dst = c.h_route[t];
        std::printf("  token %-3d -> expert_gpu %d -> y[0]=%.0f y[1]=%.0f\n",
                    t, dst, out[(size_t)t * c.hidden_dim],
                    out[(size_t)t * c.hidden_dim + 1]);
    }
}

int main(int argc, char** argv) {
    int tokens_per_rank = 2048;
    int hidden_dim = 128;
    int n = lab::require_gpus(2);
    if (n > 8) n = 8;

    if (argc > 1) tokens_per_rank = std::atoi(argv[1]);
    if (argc > 2) hidden_dim = std::atoi(argv[2]);
    if (tokens_per_rank <= 0) {
        std::fprintf(stderr, "tokens_per_rank must be > 0\n");
        return 1;
    }
    if (hidden_dim < 2) {
        std::fprintf(stderr, "hidden_dim must be >= 2 because element 0 stores token id\n");
        return 1;
    }

    std::printf("==== lesson 08: generic alltoallv dispatch/combine ====\n");
    lab::enable_all_peers(n);
    std::printf("n_gpus=%d, tokens_per_rank=%d, hidden_dim=%d\n",
                n, tokens_per_rank, hidden_dim);

    A2AContext c = setup(n, tokens_per_rank, hidden_dim);
    print_count_matrix("dispatch count[src][dst] tokens:", c.counts, n);

    float dispatch_ms = dispatch(c);
    std::printf("\n==== dispatch ====\n");
    lab::print_bandwidth("dispatch remote bytes",
                         remote_bytes(c.counts, n, hidden_dim), dispatch_ms);
    for (int r = 0; r < n; ++r) {
        std::printf("GPU%d received %d tokens for local experts\n",
                    r, col_sum(c.counts, n, r));
    }

    std::printf("\n==== local expert compute ====\n");
    run_experts(c);

    float combine_ms = combine(c);
    std::printf("\n==== combine ====\n");
    lab::print_bandwidth("combine remote bytes",
                         remote_bytes(c.combine_counts, n, hidden_dim), combine_ms);

    bool ok = verify(c);
    print_rank0_sample(c);
    std::printf("\nround trip result: %s\n", ok ? "OK" : "MISMATCH");

    teardown(c);
    std::printf("\nlesson 08 done.\n");
    return ok ? 0 : 2;
}
