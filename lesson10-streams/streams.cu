// lesson10-streams/streams.cu
//
// Make stream concurrency observable: two kernels on one stream (serial) vs
// two kernels on two streams (concurrent) vs copy+kernel overlap.

#include <cstdio>
#include <vector>

#include "checks.hpp"
#include "timing.hpp"

// Spin for roughly `target_ms` milliseconds, measured by the GPU's own clock.
// One thread, so it uses ~1 SM and leaves room for another kernel to overlap.
__global__ void busy_kernel(float target_ms) {
    unsigned long long start = clock64();
    unsigned long long ticks_needed = (unsigned long long)(target_ms * 1e6);  // ~1 GHz clock -> 1e6 ticks/ms
    while (clock64() - start < ticks_needed) { /* spin */ }
}

int main() {
    int dev = 0;
    LAB_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    LAB_CUDA(cudaGetDeviceProperties(&prop, dev));
    std::printf("==== lesson 10: CUDA streams ====\n");
    std::printf("device: %s  (%d SMs)\n\n", prop.name, prop.multiProcessorCount);

    lab::GpuTimer t;
    cudaStream_t a, b;
    LAB_CUDA(cudaStreamCreate(&a));
    LAB_CUDA(cudaStreamCreate(&b));

    // --- serial: two 40ms kernels on the same stream ---
    std::printf("[serial] two 40ms kernels on one stream\n");
    {
        t.start(a);
        busy_kernel<<<1, 1, 0, a>>>(40.0f);
        busy_kernel<<<1, 1, 0, a>>>(40.0f);
        LAB_CUDA(cudaStreamSynchronize(a));
        t.stop(a);
        std::printf("  time %.1f ms   (≈ 2×40 — no overlap)\n\n", t.elapsed_ms());
    }

    // --- concurrent: two 40ms kernels on two streams ---
    std::printf("[concurrent] two 40ms kernels on two streams\n");
    {
        t.start(a);
        busy_kernel<<<1, 1, 0, a>>>(40.0f);
        busy_kernel<<<1, 1, 0, b>>>(40.0f);
        LAB_CUDA(cudaStreamSynchronize(a));
        LAB_CUDA(cudaStreamSynchronize(b));
        t.stop(a);
        std::printf("  time %.1f ms   (≈ 40 — overlapped!)\n\n", t.elapsed_ms());
    }

    // --- copy/compute overlap: H2D on stream a, kernel on stream b ---
    std::printf("[copy/compute overlap] 64MiB H2D + 40ms kernel on separate streams\n");
    {
        size_t bytes = 1ull << 26;  // 64 MiB
        int* d;  LAB_CUDA(cudaMalloc(&d, bytes));
        int* h;  LAB_CUDA(cudaMallocHost(&h, bytes));
        for (size_t i = 0; i < bytes / sizeof(int); ++i) h[i] = (int)i;

        // baseline: H2D alone
        t.start(a);
        LAB_CUDA(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, a));
        LAB_CUDA(cudaStreamSynchronize(a));
        t.stop(a);
        float h2d_ms = t.elapsed_ms();

        // baseline: kernel alone
        t.start(b);
        busy_kernel<<<1, 1, 0, b>>>(40.0f);
        LAB_CUDA(cudaStreamSynchronize(b));
        t.stop(b);
        float kern_ms = t.elapsed_ms();

        // overlapped: both at once
        t.start(a);
        LAB_CUDA(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, a));
        busy_kernel<<<1, 1, 0, b>>>(40.0f);
        LAB_CUDA(cudaStreamSynchronize(a));
        LAB_CUDA(cudaStreamSynchronize(b));
        t.stop(a);
        float overlap_ms = t.elapsed_ms();

        std::printf("  H2D alone   %.1f ms\n", h2d_ms);
        std::printf("  kernel alone %.1f ms\n", kern_ms);
        std::printf("  overlapped  %.1f ms   (copy hid behind kernel)\n", overlap_ms);

        LAB_CUDA(cudaFree(d));
        LAB_CUDA(cudaFreeHost(h));
    }

    LAB_CUDA(cudaStreamDestroy(a));
    LAB_CUDA(cudaStreamDestroy(b));
    std::printf("\nlesson 10 done.\n");
    return 0;
}
