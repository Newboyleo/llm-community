// lesson16-moe-dispatch/moe_dispatch.cu
//
// Mini MoE dispatch: route tokens (lesson 15), AllReduce the count matrix,
// gather into per-dst buckets, then AllToAll (lesson 8) to the right GPUs.
// Host-orchestrated peer copies; lesson 17 replaces these with NVSHMEM.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "checks.hpp"
#include "multigpu.hpp"
#include "print.hpp"
#include "timing.hpp"

// ---- kernels reused from lesson 15 -----------------------------------------
__global__ void gate_kernel(const float* x, const float* W, float* logits,
                            int T, int E, int D) {
    int t = blockIdx.x;
    if (t >= T) return;
    int e = threadIdx.x;
    if (e >= E) return;
    float s = 0.f;
    for (int d = 0; d < D; ++d) s += x[t * D + d] * W[d * E + e];
    logits[t * E + e] = s;
}
__global__ void top1_kernel(const float* logits, int* assign, int T, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int best = 0; float bv = logits[t * E];
    for (int e = 1; e < E; ++e) { float v = logits[t * E + e]; if (v > bv) { bv = v; best = e; } }
    assign[t] = best;
}
// count_row[dst] += 1 for each local token whose expert lives on dst.
__global__ void count_local_kernel(const int* assign, int* count_row,
                                   int Tlocal, int n, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Tlocal) return;
    int e = assign[t];
    int dst = e / (E / n);
    atomicAdd(&count_row[dst], 1);
}

// Gather: write each token (D floats) into sendbuf at the next slot of its dst bucket.
// bucket_cursor[dst] is the running atomic write cursor; send_base[dst] the bucket start.
// We overwrite element 0 of the token with the expert id so the receiver can verify.
__global__ void gather_kernel(const float* tokens, const int* assign,
                              float* sendbuf, int* bucket_cursor, const int* send_base,
                              int Tlocal, int n, int E, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Tlocal) return;
    int e = assign[t];
    int dst = e / (E / n);
    int slot = atomicAdd(&bucket_cursor[dst], 1);
    int write_at = (send_base[dst] + slot) * D;
    for (int d = 1; d < D; ++d) sendbuf[write_at + d] = tokens[t * D + d];
    sendbuf[write_at + 0] = (float)e;  // stashed expert id for verification
}

int main(int argc, char** argv) {
    int T = 2048, E = 8, D = 256;
    int n = lab::require_gpus(2);
    if (n > 8) n = 8;
    if (argc > 1) T = std::atoi(argv[1]);
    if (argc > 2) E = std::atoi(argv[2]);
    if (argc > 3) D = std::atoi(argv[3]);
    if (T <= 0 || E <= 0 || D <= 0) {
        std::fprintf(stderr, "T, E, and D must all be positive\n");
        return 1;
    }
    if (E > 1024) {
        std::fprintf(stderr, "E must be <= 1024 because gate_kernel uses one block with E threads\n");
        return 1;
    }
    if (D < E) {
        std::fprintf(stderr, "D must be >= E for the balanced synthetic routing demo\n");
        return 1;
    }
    if (E % n != 0) { std::fprintf(stderr, "E must be divisible by n\n"); return 1; }
    if (T % n != 0) { std::fprintf(stderr, "T must be divisible by n\n"); return 1; }
    int Tlocal = T / n;

    std::printf("==== lesson 16: mini MoE dispatch ====\n");
    lab::enable_all_peers(n);
    std::printf("n=%d GPUs, T=%d tokens, E=%d experts, D=%d hidden, top-1\n", n, T, E, D);

    std::vector<float*> d_tokens(n), d_W(n), d_logits(n), d_sendbuf(n), d_recvbuf(n);
    std::vector<int*> d_assign(n), d_countrow(n), d_cursor(n), d_sendbase(n);
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMalloc(&d_tokens[r], Tlocal * D * sizeof(float)));
        LAB_CUDA(cudaMalloc(&d_W[r], D * E * sizeof(float)));
        LAB_CUDA(cudaMalloc(&d_logits[r], Tlocal * E * sizeof(float)));
        LAB_CUDA(cudaMalloc(&d_assign[r], Tlocal * sizeof(int)));
        LAB_CUDA(cudaMalloc(&d_countrow[r], n * sizeof(int)));
        LAB_CUDA(cudaMalloc(&d_cursor[r], n * sizeof(int)));
        LAB_CUDA(cudaMalloc(&d_sendbase[r], n * sizeof(int)));
        LAB_CUDA(cudaMalloc(&d_sendbuf[r], (size_t)T * D * sizeof(float)));
        LAB_CUDA(cudaMalloc(&d_recvbuf[r], (size_t)T * D * sizeof(float)));
    }

    std::vector<float> h_W(D * E, 0.f);
    for (int e = 0; e < E; ++e) h_W[e * E + e] = 1.f;
    for (int r = 0; r < n; ++r) {
        std::vector<float> ht(Tlocal * D, 0.f);
        for (int t = 0; t < Tlocal; ++t) {
            int expert = (r * Tlocal + t) % E;
            ht[t * D + expert] = 1.f;
        }
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMemcpy(d_tokens[r], ht.data(), ht.size() * sizeof(float), cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(d_W[r], h_W.data(), h_W.size() * sizeof(float), cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemset(d_countrow[r], 0, n * sizeof(int)));
        LAB_CUDA(cudaMemset(d_cursor[r], 0, n * sizeof(int)));
    }

    // per-GPU: gate, top1, count_local
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        gate_kernel<<<Tlocal, E>>>(d_tokens[r], d_W[r], d_logits[r], Tlocal, E, D);
        top1_kernel<<<(Tlocal + 255) / 256, 256>>>(d_logits[r], d_assign[r], Tlocal, E);
        count_local_kernel<<<(Tlocal + 255) / 256, 256>>>(d_assign[r], d_countrow[r], Tlocal, n, E);
        LAB_CUDA_SYNC();
    }

    // AllReduce count (naive, on host): assemble global_count[n][n]
    std::vector<int> global_count(n * n, 0);
    for (int r = 0; r < n; ++r) {
        std::vector<int> row(n);
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMemcpy(row.data(), d_countrow[r], n * sizeof(int), cudaMemcpyDeviceToHost));
        for (int d = 0; d < n; ++d) global_count[r * n + d] = row[d];
    }
    std::vector<int> send_base(n * n, 0), recv_offset(n * n, 0);
    for (int s = 0; s < n; ++s) { int acc = 0; for (int d = 0; d < n; ++d) { send_base[s * n + d] = acc; acc += global_count[s * n + d]; } }
    for (int d = 0; d < n; ++d) { int acc = 0; for (int s = 0; s < n; ++s) { recv_offset[s * n + d] = acc; acc += global_count[s * n + d]; } }

    std::printf("\nglobal count[src][dst]:\n       ");
    for (int d = 0; d < n; ++d) std::printf("dst%-4d", d);
    std::printf("\n");
    for (int s = 0; s < n; ++s) {
        std::printf("src%-3d ", s);
        for (int d = 0; d < n; ++d) std::printf("%-7d", global_count[s * n + d]);
        std::printf("\n");
    }

    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMemcpy(d_sendbase[r], &send_base[r * n], n * sizeof(int), cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemset(d_cursor[r], 0, n * sizeof(int)));
    }

    // gather
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        gather_kernel<<<(Tlocal + 255) / 256, 256>>>(d_tokens[r], d_assign[r],
            d_sendbuf[r], d_cursor[r], d_sendbase[r], Tlocal, n, E, D);
        LAB_CUDA_SYNC();
    }

    // AllToAll
    std::vector<cudaStream_t> streams(n);
    for (int r = 0; r < n; ++r) { LAB_CUDA(cudaSetDevice(r)); LAB_CUDA(cudaStreamCreate(&streams[r])); }
    for (int src = 0; src < n; ++src) {
        for (int dst = 0; dst < n; ++dst) {
            int cnt = global_count[src * n + dst];
            if (cnt == 0) continue;
            size_t bytes = (size_t)cnt * D * sizeof(float);
            int s_off = send_base[src * n + dst];
            int d_off = recv_offset[src * n + dst];
            LAB_CUDA(cudaSetDevice(src));
            LAB_CUDA(cudaMemcpyPeerAsync(d_recvbuf[dst] + (size_t)d_off * D, dst,
                                         d_sendbuf[src] + (size_t)s_off * D, src,
                                         bytes, streams[src]));
        }
    }
    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaStreamSynchronize(streams[r]));
    }

    // verify: each received token's stashed expert id maps to the receiving GPU
    bool ok = true;
    int experts_per_gpu = E / n;
    for (int r = 0; r < n; ++r) {
        int total = 0; for (int s = 0; s < n; ++s) total += global_count[s * n + r];
        std::vector<float> host((size_t)total * D);
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaMemcpy(host.data(), d_recvbuf[r], host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int t = 0; t < total; ++t) {
            int e = (int)host[t * D];
            if (e / experts_per_gpu != r) { ok = false; break; }
        }
        std::printf("GPU%d received %d tokens (experts %d..%d live here)\n",
                    r, total, r * experts_per_gpu, r * experts_per_gpu + experts_per_gpu - 1);
    }
    std::printf("all tokens routed to the correct GPU: %s\n", ok ? "YES" : "NO");

    for (int r = 0; r < n; ++r) {
        LAB_CUDA(cudaSetDevice(r));
        LAB_CUDA(cudaFree(d_tokens[r])); LAB_CUDA(cudaFree(d_W[r])); LAB_CUDA(cudaFree(d_logits[r]));
        LAB_CUDA(cudaFree(d_assign[r])); LAB_CUDA(cudaFree(d_countrow[r])); LAB_CUDA(cudaFree(d_cursor[r]));
        LAB_CUDA(cudaFree(d_sendbase[r])); LAB_CUDA(cudaFree(d_sendbuf[r])); LAB_CUDA(cudaFree(d_recvbuf[r]));
        LAB_CUDA(cudaStreamDestroy(streams[r]));
    }
    std::printf("\nlesson 16 done.\n");
    return 0;
}
