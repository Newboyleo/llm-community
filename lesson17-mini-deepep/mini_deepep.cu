// lesson17-mini-deepep/mini_deepep.cu
//
// Mini DeepEP dispatch: device-side NVSHMEM puts into symmetric per-channel
// receive buffers, per-channel ready flags (quiet-guarded), multiple channels
// in parallel. This is the structural skeleton of DeepEP's intranode_dispatch.
//
// Simplifications vs real DeepEP (called out in the README):
//   - top-1 routing (real DeepEP: top-k)
//   - receive layout is per-source slotted (src writes to recvbuf[src*Tlocal..])
//     rather than prefix-sum-packed; correct and simple, slightly wasteful.
//   - no FP8 quantization (lesson 19 adds it conceptually)
//   - gather is fused into the dispatch kernel (one put per token), like DeepEP.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "checks.hpp"
#include "print.hpp"

// ---- routing kernels (lesson 15) ------------------------------------------
__global__ void gate_kernel(const float* x, const float* W, float* logits, int T, int E, int D) {
    int t = blockIdx.x; if (t >= T) return;
    int e = threadIdx.x; if (e >= E) return;
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
__global__ void count_local_kernel(const int* assign, int* count_row, int Tlocal, int n, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Tlocal) return;
    int e = assign[t]; int dst = e / (E / n);
    atomicAdd(&count_row[dst], 1);
}

// ---- dispatch kernel: one block per (channel, dst) -------------------------
// Each block walks this PE's local tokens, finds those destined for `dst`, and
// (for its channel's share) puts them into dst's recvbuf at slot
//   src*Tlocal + running_index
// putting one float at a time (D floats per token). Then quiet + flag put.
template <int D>
__global__ void dispatch_kernel(const float* __restrict__ tokens,
                                const int* __restrict__ assign,
                                float* __restrict__ recvbuf,   // symmetric, [n*Tlocal*D]
                                int* __restrict__ ready,       // symmetric, [nch*n]
                                int Tlocal, int n, int E, int nch, int src_pe) {
    int ch = blockIdx.x;
    int dst = blockIdx.y;
    if (ch >= nch || dst >= n) return;

    int experts_per_gpu = E / n;
    if (threadIdx.x != 0) return;
    int local_idx = 0;  // running count of tokens for dst seen by this block

    for (int t = 0; t < Tlocal; ++t) {
        int e = assign[t];
        if (e / experts_per_gpu != dst) continue;
        if (local_idx % nch != ch) { ++local_idx; continue; }
        // slot in dst's recvbuf: this src's region starts at src_pe*Tlocal
        size_t slot = (size_t)src_pe * Tlocal + local_idx;
        for (int d = 0; d < D; ++d) {
            float v = (d == 0) ? (float)(e + 1) : tokens[t * D + d];
            nvshmem_float_put(&recvbuf[slot * D + d], &v, 1, dst);
        }
        ++local_idx;
    }
    nvshmem_quiet();  // data must land before the flag
    int seq = 1;
    nvshmem_int_put(&ready[ch * n + src_pe], &seq, 1, dst);
    nvshmem_quiet();
}

// Consumer: poll every (channel, src) ready flag for this PE; set arrived.
__global__ void consume_kernel(int* ready, int* arrived, int n, int nch) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int s = 0; s < n; ++s)
        for (int c = 0; c < nch; ++c) {
            nvshmem_int_wait_until(ready + c * n + s, NVSHMEM_CMP_EQ, 1);
        }
    *arrived = 1;
}

int main(int argc, char** argv) {
    int T = 2048, E = 8, D = 256, nch = 4;
    nvshmem_init();
    int n = nvshmem_n_pes();
    int pe = nvshmem_my_pe();
    if (n < 2) { if (pe == 0) std::fprintf(stderr, "needs >=2 PEs\n"); nvshmem_finalize(); return 1; }
    LAB_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
    if (argc > 1) T = std::atoi(argv[1]);
    if (argc > 2) E = std::atoi(argv[2]);
    if (argc > 3) D = std::atoi(argv[3]);
    if (argc > 4) nch = std::atoi(argv[4]);
    if (T <= 0 || E <= 0 || D <= 0) {
        if (pe == 0) std::fprintf(stderr, "T, E, and D must all be positive\n");
        nvshmem_finalize();
        return 1;
    }
    if (E > 1024) {
        if (pe == 0)
            std::fprintf(stderr, "E must be <= 1024 because gate_kernel uses one block with E threads\n");
        nvshmem_finalize();
        return 1;
    }
    if (E % n != 0) { if (nvshmem_my_pe() == 0) std::fprintf(stderr, "E must be divisible by n\n"); nvshmem_finalize(); return 1; }
    if (T % n != 0) { if (nvshmem_my_pe() == 0) std::fprintf(stderr, "T must be divisible by n\n"); nvshmem_finalize(); return 1; }
    if (D != 64 && D != 128 && D != 256) {
        if (nvshmem_my_pe() == 0) std::fprintf(stderr, "D must be one of 64, 128, 256\n");
        nvshmem_finalize();
        return 1;
    }
    if (nch <= 0) { if (nvshmem_my_pe() == 0) std::fprintf(stderr, "channels must be > 0\n"); nvshmem_finalize(); return 1; }
    int Tlocal = T / n;

    if (pe == 0)
        std::printf("==== lesson 17: mini DeepEP dispatch ====\n"
                    "n=%d PEs, T=%d, E=%d, D=%d, channels=%d\n", n, T, E, D, nch);

    // symmetric allocations. recvbuf is sized n*Tlocal*D so every src has a region.
    float* tokens  = (float*)nvshmem_malloc((size_t)Tlocal * D * sizeof(float));
    float* W       = (float*)nvshmem_malloc((size_t)D * E * sizeof(float));
    float* logits  = (float*)nvshmem_malloc((size_t)Tlocal * E * sizeof(float));
    int* assign    = (int*)nvshmem_malloc((size_t)Tlocal * sizeof(int));
    int* count_row = (int*)nvshmem_malloc(n * sizeof(int));
    float* recvbuf = (float*)nvshmem_malloc((size_t)n * Tlocal * D * sizeof(float));
    int* ready     = (int*)nvshmem_malloc((size_t)nch * n * sizeof(int));
    int* arrived   = (int*)nvshmem_malloc(sizeof(int));

    {
        std::vector<float> ht((size_t)Tlocal * D);
        for (int i = 0; i < Tlocal * D; ++i) ht[i] = (float)(pe * 1000 + (i & 0xff));
        LAB_CUDA(cudaMemcpy(tokens, ht.data(), ht.size() * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> hw((size_t)D * E);
        for (int i = 0; i < D * E; ++i) hw[i] = (float)((i * 214013 + 2531011) & 0xff) / 256.f - 0.5f;
        LAB_CUDA(cudaMemcpy(W, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemset(count_row, 0, n * sizeof(int)));
        LAB_CUDA(cudaMemset(recvbuf, 0, (size_t)n * Tlocal * D * sizeof(float)));
        LAB_CUDA(cudaMemset(ready, 0, (size_t)nch * n * sizeof(int)));
        LAB_CUDA(cudaMemset(arrived, 0, sizeof(int)));
    }
    nvshmem_barrier_all();

    // routing
    gate_kernel<<<Tlocal, E>>>(tokens, W, logits, Tlocal, E, D);
    top1_kernel<<<(Tlocal + 255) / 256, 256>>>(logits, assign, Tlocal, E);
    count_local_kernel<<<(Tlocal + 255) / 256, 256>>>(assign, count_row, Tlocal, n, E);
    LAB_CUDA_SYNC();
    nvshmem_barrier_all();

    // gather the global count matrix via one-sided gets (naive AllReduce-sum, host-side)
    std::vector<int> global(n * n, 0);
    for (int r = 0; r < n; ++r)
        for (int d = 0; d < n; ++d)
            global[r * n + d] = nvshmem_int_g(&count_row[d], r);

    if (pe == 0) {
        std::printf("\nglobal count[src][dst]:\n       ");
        for (int d = 0; d < n; ++d) std::printf("dst%-4d", d);
        std::printf("\n");
        for (int s = 0; s < n; ++s) {
            std::printf("src%-3d ", s);
            for (int d = 0; d < n; ++d) std::printf("%-7d", global[s * n + d]);
            std::printf("\n");
        }
    }

    // dispatch: one block per (channel, dst)
    cudaStream_t s; LAB_CUDA(cudaStreamCreate(&s));
    dim3 grid(nch, n);
    if (D == 256)      dispatch_kernel<256><<<grid, 64, 0, s>>>(tokens, assign, recvbuf, ready, Tlocal, n, E, nch, pe);
    else if (D == 128) dispatch_kernel<128><<<grid, 64, 0, s>>>(tokens, assign, recvbuf, ready, Tlocal, n, E, nch, pe);
    else               dispatch_kernel<64><<<grid, 64, 0, s>>>(tokens, assign, recvbuf, ready, Tlocal, n, E, nch, pe);
    LAB_CUDA_SYNC();

    // consume: poll all (channel, src) flags for this PE
    consume_kernel<<<1, 1, 0, s>>>(ready, arrived, n, nch);
    LAB_CUDA_SYNC();
    nvshmem_barrier_all();

    // verify: every token written into this PE's recvbuf has expert id mapping here
    int experts_per_gpu = E / n;
    int total_incoming = 0;
    for (int s = 0; s < n; ++s) total_incoming += global[s * n + pe];
    std::vector<float> host((size_t)n * Tlocal * D);
    LAB_CUDA(cudaMemcpy(host.data(), recvbuf, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    int seen = 0; bool ok = true;
    for (int s = 0; s < n && ok; ++s) {
        for (int i = 0; i < Tlocal; ++i) {
            float e_f = host[((size_t)s * Tlocal + i) * D];
            if (e_f == 0.f) continue;  // empty slot (no token from src s here)
            int e = (int)e_f - 1;
            if (e / experts_per_gpu != pe) { ok = false; break; }
            ++seen;
        }
    }
    if (seen != total_incoming) ok = false;
    std::printf("PE%d: %d tokens arrived, all map here: %s\n", pe, seen, ok ? "YES" : "NO");

    LAB_CUDA(cudaStreamDestroy(s));
    nvshmem_free(tokens); nvshmem_free(W); nvshmem_free(logits); nvshmem_free(assign);
    nvshmem_free(count_row); nvshmem_free(recvbuf); nvshmem_free(ready); nvshmem_free(arrived);
    nvshmem_finalize();
    if (pe == 0) std::printf("\nlesson 17 done.\n");
    return 0;
}
