// lesson15-token-routing/token_routing.cu
//
// The routing stage of a MoE layer: gate matmul -> top-1 -> count[src][dst]
// matrix -> prefix-sum offsets. Single GPU; the output (count + offsets) is
// the dispatch plan that lessons 16/17 consume across GPUs.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "checks.hpp"
#include "print.hpp"

// logits[t,e] = sum_d x[t,d] * W[d,e]
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

// assign[t] = argmax_e logits[t,e]
__global__ void top1_kernel(const float* logits, int* assign, int T, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int best = 0;
    float bv = logits[t * E + 0];
    for (int e = 1; e < E; ++e) {
        float v = logits[t * E + e];
        if (v > bv) { bv = v; best = e; }
    }
    assign[t] = best;
}

// count[src][dst] += 1 for each token. src = t/(T/n), dst = e/(E/n).
__global__ void count_kernel(const int* assign, int* count, int T, int n, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int e = assign[t];
    int src = (t * n) / T;
    int dst = e / (E / n);
    atomicAdd(&count[src * n + dst], 1);
}

// offsets[src][dst] = sum_{s<src} count[s][dst]  (column prefix sum)
__global__ void prefix_kernel(const int* count, int* offsets, int n) {
    // one thread does it all; n is small (<=8).
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int dst = 0; dst < n; ++dst) {
        int acc = 0;
        for (int src = 0; src < n; ++src) {
            offsets[src * n + dst] = acc;
            acc += count[src * n + dst];
        }
    }
}

int main(int argc, char** argv) {
    int T = 1024, E = 8, D = 256, n = 4;
    if (argc > 1) T = std::atoi(argv[1]);
    if (argc > 2) E = std::atoi(argv[2]);
    if (argc > 3) D = std::atoi(argv[3]);
    if (argc > 4) n = std::atoi(argv[4]);
    if (T <= 0 || E <= 0 || D <= 0 || n <= 0) {
        std::fprintf(stderr, "T, E, D, and n must all be positive\n");
        return 1;
    }
    if (E > 1024) {
        std::fprintf(stderr, "E must be <= 1024 because gate_kernel uses one block with E threads\n");
        return 1;
    }
    if (E % n != 0) { std::fprintf(stderr, "E must be divisible by n\n"); return 1; }

    std::printf("==== lesson 15: token routing ====\n");
    std::printf("T=%d tokens, E=%d experts, D=%d hidden, n=%d GPUs (E/n=%d experts/GPU)\n\n",
                T, E, D, n, E / n);

    // random-ish gate weights and tokens (deterministic, not uniform, for a
    // visible but non-trivial routing)
    std::vector<float> h_x(T * D), h_W(D * E);
    for (int i = 0; i < T * D; ++i) h_x[i] = (float)((i * 1103515245 + 12345) & 0xff) / 256.f;
    for (int i = 0; i < D * E; ++i) h_W[i] = (float)((i * 214013 + 2531011) & 0xff) / 256.f - 0.5f;

    float *d_x, *d_W, *d_logits;
    int *d_assign, *d_count, *d_offsets;
    LAB_CUDA(cudaMalloc(&d_x, T * D * sizeof(float)));
    LAB_CUDA(cudaMalloc(&d_W, D * E * sizeof(float)));
    LAB_CUDA(cudaMalloc(&d_logits, T * E * sizeof(float)));
    LAB_CUDA(cudaMalloc(&d_assign, T * sizeof(int)));
    LAB_CUDA(cudaMalloc(&d_count, n * n * sizeof(int)));
    LAB_CUDA(cudaMalloc(&d_offsets, n * n * sizeof(int)));
    LAB_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    LAB_CUDA(cudaMemcpy(d_W, h_W.data(), h_W.size() * sizeof(float), cudaMemcpyHostToDevice));
    LAB_CUDA(cudaMemset(d_count, 0, n * n * sizeof(int)));

    gate_kernel<<<T, E>>>(d_x, d_W, d_logits, T, E, D);
    LAB_CUDA_SYNC();
    top1_kernel<<<(T + 255) / 256, 256>>>(d_logits, d_assign, T, E);
    LAB_CUDA_SYNC();
    count_kernel<<<(T + 255) / 256, 256>>>(d_assign, d_count, T, n, E);
    LAB_CUDA_SYNC();
    prefix_kernel<<<1, 1>>>(d_count, d_offsets, n);
    LAB_CUDA_SYNC();

    std::vector<int> assign(T), count(n * n), offsets(n * n);
    LAB_CUDA(cudaMemcpy(assign.data(), d_assign, T * sizeof(int), cudaMemcpyDeviceToHost));
    LAB_CUDA(cudaMemcpy(count.data(), d_count, n * n * sizeof(int), cudaMemcpyDeviceToHost));
    LAB_CUDA(cudaMemcpy(offsets.data(), d_offsets, n * n * sizeof(int), cudaMemcpyDeviceToHost));

    lab::print_host("assign", assign.data(), 8, 8);
    std::printf("\ncount[src][dst]  (rows=src GPU, cols=dst GPU):\n       ");
    for (int d = 0; d < n; ++d) std::printf("dst%-4d", d);
    std::printf("\n");
    for (int s = 0; s < n; ++s) {
        std::printf("src%-3d ", s);
        for (int d = 0; d < n; ++d) std::printf("%-7d", count[s * n + d]);
        std::printf("\n");
    }
    std::printf("\noffsets[src][dst] = prefix sum of count[*][dst] over src:\n       ");
    for (int d = 0; d < n; ++d) std::printf("dst%-4d", d);
    std::printf("\n");
    for (int s = 0; s < n; ++s) {
        std::printf("src%-3d ", s);
        for (int d = 0; d < n; ++d) std::printf("%-7d", offsets[s * n + d]);
        std::printf("\n");
    }

    LAB_CUDA(cudaFree(d_x)); LAB_CUDA(cudaFree(d_W)); LAB_CUDA(cudaFree(d_logits));
    LAB_CUDA(cudaFree(d_assign)); LAB_CUDA(cudaFree(d_count)); LAB_CUDA(cudaFree(d_offsets));
    std::printf("\nlesson 15 done.\n");
    return 0;
}
