// lesson01-gpu-copy/gpu_copy.cu
//
// The simplest possible GPU program: allocate on host, copy to device, run a
// kernel, copy back. We also measure H2D / D2H bandwidth for both PAGEABLE and
// PINNED host memory so you can see the difference — that difference is the
// lesson.

#include <cstdio>
#include <vector>

#include "checks.hpp"
#include "print.hpp"
#include "timing.hpp"

// One thread per element: x[i] *= 2.
template <typename T>
__global__ void double_inplace(T* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2;
}

// Fill 1,2,3,...
static void init_host(int* h, int n) {
    for (int i = 0; i < n; ++i) h[i] = i + 1;
}

int main(int argc, char** argv) {
    // --- pick a size ---------------------------------------------------------
    int n = 1 << 24;  // 16M ints = 64 MiB
    if (argc > 1) n = std::atoi(argv[1]);
    if (n <= 0) n = 1 << 24;

    int dev = 0;
    LAB_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    LAB_CUDA(cudaGetDeviceProperties(&prop, dev));
    std::printf("==== lesson 01: GPU copy ====\n");
    std::printf("device %d: %s  (compute %d)\n", dev, prop.name, prop.major * 10 + prop.minor);
    std::printf("n = %d ints (%.2f MiB)\n\n", n, n * sizeof(int) / (1024.0 * 1024.0));

    // --- correctness round trip ---------------------------------------------
    std::vector<int> h_x(n), h_y(n);
    init_host(h_x.data(), n);

    int* d_x;
    LAB_CUDA(cudaMalloc(&d_x, n * sizeof(int)));
    LAB_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (n + 255) / 256;
    double_inplace<int><<<blocks, 256>>>(d_x, n);
    LAB_CUDA_SYNC();

    LAB_CUDA(cudaMemcpy(h_y.data(), d_x, n * sizeof(int), cudaMemcpyDeviceToHost));

    lab::banner("correctness");
    lab::print_host("h_x", h_x.data(), n, 4);
    lab::print_host("h_y", h_y.data(), n, 4);
    bool ok = true;
    for (int i = 0; i < n; ++i)
        if (h_y[i] != 2 * h_x[i]) { ok = false; break; }
    std::printf("%s\n", ok ? "OK" : "MISMATCH");

    // --- bandwidth: PAGEABLE host memory ------------------------------------
    // std::vector backing is ordinary page-able memory. The driver has to
    // stage it through pinned memory before DMA, which costs a host memcpy.
    lab::banner("timing: PAGEABLE host memory");
    {
        lab::GpuTimer t;
        std::vector<int> tmp(n);
        init_host(tmp.data(), n);
        size_t bytes = n * sizeof(int);

        t.start();
        LAB_CUDA(cudaMemcpy(d_x, tmp.data(), bytes, cudaMemcpyHostToDevice));
        t.stop();
        lab::print_bandwidth("H2D pageable", bytes, t.elapsed_ms());

        t.start();
        LAB_CUDA(cudaMemcpy(tmp.data(), d_x, bytes, cudaMemcpyDeviceToHost));
        t.stop();
        lab::print_bandwidth("D2H pageable", bytes, t.elapsed_ms());
    }

    // --- bandwidth: PINNED host memory --------------------------------------
    // cudaMallocHost gives the driver a pinned, DMA-able buffer. No staging,
    // no extra host memcpy. This is the real PCIe bandwidth.
    lab::banner("timing: PINNED host memory");
    {
        int* h_pin = nullptr;
        LAB_CUDA(cudaMallocHost(&h_pin, n * sizeof(int)));
        init_host(h_pin, n);
        size_t bytes = n * sizeof(int);
        lab::GpuTimer t;

        t.start();
        LAB_CUDA(cudaMemcpy(d_x, h_pin, bytes, cudaMemcpyHostToDevice));
        t.stop();
        lab::print_bandwidth("H2D pinned", bytes, t.elapsed_ms());

        t.start();
        LAB_CUDA(cudaMemcpy(h_pin, d_x, bytes, cudaMemcpyDeviceToHost));
        t.stop();
        lab::print_bandwidth("D2H pinned", bytes, t.elapsed_ms());

        LAB_CUDA(cudaFreeHost(h_pin));
    }

    // --- bandwidth: D2D on the SAME device (HBM ceiling) --------------------
    lab::banner("timing: D2D same device (HBM)");
    {
        int* d_y;
        LAB_CUDA(cudaMalloc(&d_y, n * sizeof(int)));
        size_t bytes = n * sizeof(int);
        lab::GpuTimer t;
        t.start();
        LAB_CUDA(cudaMemcpy(d_y, d_x, bytes, cudaMemcpyDeviceToDevice));
        t.stop();
        lab::print_bandwidth("D2D same-dev", bytes, t.elapsed_ms());
        LAB_CUDA(cudaFree(d_y));
    }

    LAB_CUDA(cudaFree(d_x));
    std::printf("\nlesson 01 done.\n");
    return 0;
}
