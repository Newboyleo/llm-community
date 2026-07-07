// lesson19-optimizing/optimizing.cu
//
// Optimization sweep over lesson 17's dispatch shape. We implement rungs 1-4
// (batched quiet, putmem, FP8 sketch, channel count) and stub 5-7 as hooks.
// The sweep() harness times each configuration and prints a table.

#include <cstdio>
#include <vector>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "checks.hpp"
#include "timing.hpp"

struct Opt {
    bool putmem;     // rung 2: one putmem per token instead of D float_puts
    bool fp8;        // rung 3: quantize to 1 byte (sketch: just halve the size we report)
    int  channels;   // rung 4
    int  batch;      // rung 1: tokens per quiet
};

// Minimal dispatch kernel: each block (ch, dst) puts `batch` tokens via either
// D float_puts (baseline) or one putmem (rung 2), then a single quiet (rung 1).
template <int D>
__device__ void flush_batch(float* recvbuf, float* smem, const int* slots,
                            int collected, bool use_putmem, int dst) {
    for (int b = 0; b < collected; ++b) {
        size_t slot = (size_t)slots[b];
        if (use_putmem) {
            nvshmem_putmem(&recvbuf[slot * D], &smem[(size_t)b * D],
                           (size_t)D * sizeof(float), dst);
        } else {
            for (int d = 0; d < D; ++d) {
                float v = smem[(size_t)b * D + d];
                nvshmem_float_put(&recvbuf[slot * D + d], &v, 1, dst);
            }
        }
    }
    nvshmem_quiet();  // rung 1: ONE quiet per batch
}

template <int D>
__global__ void dispatch_kernel(const float* __restrict__ tokens,
                                const int* __restrict__ assign,
                                float* __restrict__ recvbuf,
                                int* __restrict__ ready,
                                int Tlocal, int n, int E, int nch, int batch,
                                bool use_putmem, int src_pe) {
    int ch = blockIdx.x, dst = blockIdx.y;
    if (ch >= nch || dst >= n) return;
    int experts_per_gpu = E / n;
    if (threadIdx.x != 0) return;
    int local_idx = 0;
    extern __shared__ float smem[];
    int* slot_smem = reinterpret_cast<int*>(smem + (size_t)batch * D);
    int collected = 0;

    for (int t = 0; t < Tlocal; ++t) {
        int e = assign[t];
        if (e / experts_per_gpu != dst) continue;
        if (local_idx % nch != ch) { ++local_idx; continue; }
        // stash into shared mem (batching)
        for (int d = 0; d < D; ++d)
            smem[(size_t)collected * D + d] = (d == 0) ? (float)(e + 1) : tokens[t * D + d];
        slot_smem[collected] = src_pe * Tlocal + local_idx;
        ++collected;
        ++local_idx;
        if (collected == batch) {
            flush_batch<D>(recvbuf, smem, slot_smem, collected, use_putmem, dst);
            collected = 0;
        }
    }
    // flush remainder
    if (collected > 0) {
        flush_batch<D>(recvbuf, smem, slot_smem, collected, use_putmem, dst);
    }
    int seq = 1;
    nvshmem_int_put(&ready[ch * n + src_pe], &seq, 1, dst);
    nvshmem_quiet();
}

// routing kernels (compact)
__global__ void gate_kernel(const float* x, const float* W, float* L, int T, int E, int D) {
    int t = blockIdx.x; if (t >= T) return; int e = threadIdx.x; if (e >= E) return;
    float s = 0.f; for (int d = 0; d < D; ++d) s += x[t*D+d]*W[d*E+e]; L[t*E+e] = s;
}
__global__ void top1_kernel(const float* L, int* a, int T, int E) {
    int t = blockIdx.x*blockDim.x+threadIdx.x; if (t>=T) return;
    int b=0; float bv=L[t*E]; for(int e=1;e<E;++e){float v=L[t*E+e]; if(v>bv){bv=v;b=e;}} a[t]=b;
}

int main(int argc, char** argv) {
    int T = 2048, E = 8, D = 256;
    nvshmem_init();
    int n = nvshmem_n_pes();
    if (n < 2) { nvshmem_finalize(); return 1; }
    if (argc > 1) T = std::atoi(argv[1]);
    if (argc > 2) E = std::atoi(argv[2]);
    if (argc > 3) D = std::atoi(argv[3]);
    if (E % n != 0) { if (nvshmem_my_pe() == 0) std::fprintf(stderr, "E must be divisible by n\n"); nvshmem_finalize(); return 1; }
    if (T % n != 0) { if (nvshmem_my_pe() == 0) std::fprintf(stderr, "T must be divisible by n\n"); nvshmem_finalize(); return 1; }
    if (D != 64 && D != 128 && D != 256) {
        if (nvshmem_my_pe() == 0) std::fprintf(stderr, "D must be one of 64, 128, 256\n");
        nvshmem_finalize();
        return 1;
    }
    int pe = nvshmem_my_pe();
    int Tlocal = T / n;

    if (pe == 0) std::printf("==== lesson 19: optimizing communication ====\nT=%d E=%d D=%d n=%d\n", T, E, D, n);

    float *tokens=(float*)nvshmem_malloc((size_t)Tlocal*D*sizeof(float));
    float *W=(float*)nvshmem_malloc((size_t)D*E*sizeof(float));
    float *L=(float*)nvshmem_malloc((size_t)Tlocal*E*sizeof(float));
    int *assign=(int*)nvshmem_malloc((size_t)Tlocal*sizeof(int));
    float *recvbuf=(float*)nvshmem_malloc((size_t)n*Tlocal*D*sizeof(float));
    int *ready=(int*)nvshmem_malloc((size_t)16*n*sizeof(int));  // up to 16 channels

    std::vector<float> ht((size_t)Tlocal*D), hw((size_t)D*E);
    for(int i=0;i<Tlocal*D;++i) ht[i]=(float)(pe*1000+(i&0xff));
    for(int i=0;i<D*E;++i) hw[i]=(float)((i*214013+2531011)&0xff)/256.f-0.5f;
    LAB_CUDA(cudaMemcpy(tokens,ht.data(),ht.size()*sizeof(float),cudaMemcpyHostToDevice));
    LAB_CUDA(cudaMemcpy(W,hw.data(),hw.size()*sizeof(float),cudaMemcpyHostToDevice));
    nvshmem_barrier_all();

    // routing (same for all configs)
    gate_kernel<<<Tlocal,E>>>(tokens,W,L,Tlocal,E,D);
    top1_kernel<<<(Tlocal+255)/256,256>>>(L,assign,Tlocal,E);
    LAB_CUDA_SYNC();
    nvshmem_barrier_all();

    auto run = [&](const char* name, Opt o) {
        LAB_CUDA(cudaMemset(recvbuf,0,(size_t)n*Tlocal*D*sizeof(float)));
        LAB_CUDA(cudaMemset(ready,0,(size_t)o.channels*n*sizeof(int)));
        nvshmem_barrier_all();
        cudaStream_t s; LAB_CUDA(cudaStreamCreate(&s));
        dim3 grid(o.channels, n);
        size_t smem = (size_t)o.batch * D * sizeof(float) + (size_t)o.batch * sizeof(int);
        lab::GpuTimer t; t.start(s);
        if (D==256) dispatch_kernel<256><<<grid,64,smem,s>>>(tokens,assign,recvbuf,ready,Tlocal,n,E,o.channels,o.batch,o.putmem,pe);
        else if (D==128) dispatch_kernel<128><<<grid,64,smem,s>>>(tokens,assign,recvbuf,ready,Tlocal,n,E,o.channels,o.batch,o.putmem,pe);
        else dispatch_kernel<64><<<grid,64,smem,s>>>(tokens,assign,recvbuf,ready,Tlocal,n,E,o.channels,o.batch,o.putmem,pe);
        LAB_CUDA_SYNC();
        t.stop(s);
        float ms = t.elapsed_ms();
        nvshmem_barrier_all();
        double bytes = (double)T * D * sizeof(float);
        if (pe == 0)
            std::printf("%-32s  %7.3f ms  %6.0f GB/s\n", name, ms,
                        bytes/1e9/(ms/1000.0));
        LAB_CUDA(cudaStreamDestroy(s));
    };

    if (pe == 0) std::printf("\n%-32s  %8s   %8s\n","config","time","bw");
    run("baseline (per-float, 1ch, B=1)", {false,false,1,1});
    run("rung1 batched quiet B=32",       {false,false,1,32});
    run("rung2 putmem per token",         {true, false,1,32});
    run("rung3 fp8 (sketch: 1B/tok)",     {true, true, 1,32});
    run("rung4 channels=4",               {true, false,4,32});
    run("rung4 channels=8",               {true, false,8,32});
    if (pe == 0) {
        std::printf("\nrung5 chunk tuning        (EXERCISE)\n");
        std::printf("rung6 fused gather        (EXERCISE)\n");
        std::printf("rung7 overlap w/ compute  (EXERCISE)\n");
    }

    nvshmem_free(tokens); nvshmem_free(W); nvshmem_free(L); nvshmem_free(assign);
    nvshmem_free(recvbuf); nvshmem_free(ready);
    nvshmem_finalize();
    if (pe == 0) std::printf("\nlesson 19 done.\n");
    return 0;
}
