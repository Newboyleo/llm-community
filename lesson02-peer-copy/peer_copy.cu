// lesson02-peer-copy/peer_copy.cu
//
// Copy a buffer from GPU0's HBM to GPU1's HBM three ways:
//   1. naive  : GPU0 -> host -> GPU1   (two PCIe trips)
//   2. peer   : cudaMemcpyPeer         (NVLink, requires peer access enabled)
//   3. UVA    : plain cudaMemcpy D2D   (same physical path, runtime infers devs)
//
// The point is to see the bandwidth gap and to learn that forgetting to enable
// peer access silently falls back to the slow path.

#include <cstdio>
#include <vector>

#include "checks.hpp"
#include "multigpu.hpp"
#include "print.hpp"
#include "timing.hpp"

// Time one copy of `bytes` from dev_src to dev_dst. If through_host, force the
// naive bounce path.
static float bench_one(int dev_src, int dev_dst, size_t bytes, bool through_host,
                       const char* label) {
    // Allocate source on dev_src, dest on dev_dst.
    int* d_src = nullptr;
    int* d_dst = nullptr;
    LAB_CUDA(cudaSetDevice(dev_src));
    LAB_CUDA(cudaMalloc(&d_src, bytes));
    LAB_CUDA(cudaMemset(d_src, 0xAB, bytes));  // arbitrary fill
    LAB_CUDA(cudaSetDevice(dev_dst));
    LAB_CUDA(cudaMalloc(&d_dst, bytes));
    LAB_CUDA(cudaMemset(d_dst, 0, bytes));

    // Pinned host staging buffer for the naive path.
    int* h_stage = nullptr;
    LAB_CUDA(cudaMallocHost(&h_stage, bytes));

    float ms = 0.0f;
    if (through_host) {
        // This path spans two devices plus host staging, so CPU wall-clock is
        // the honest end-to-end measurement.
        lab::CpuTimer t;
        t.start();
        LAB_CUDA(cudaSetDevice(dev_src));
        LAB_CUDA(cudaMemcpy(h_stage, d_src, bytes, cudaMemcpyDeviceToHost));
        LAB_CUDA(cudaSetDevice(dev_dst));
        LAB_CUDA(cudaMemcpy(d_dst, h_stage, bytes, cudaMemcpyHostToDevice));
        ms = static_cast<float>(t.elapsed_ms());
    } else {
        // Peer path. Make sure peer access is enabled both directions.
        lab::enable_all_peers(2);
        LAB_CUDA(cudaSetDevice(dev_dst));
        cudaStream_t s;
        LAB_CUDA(cudaStreamCreate(&s));
        lab::GpuTimer t;
        t.start(s);
        LAB_CUDA(cudaMemcpyPeerAsync(d_dst, dev_dst, d_src, dev_src, bytes, s));
        t.stop(s);
        ms = t.elapsed_ms();
        LAB_CUDA(cudaStreamDestroy(s));
    }
    lab::print_bandwidth(label, bytes, ms);

    LAB_CUDA(cudaSetDevice(dev_src));
    LAB_CUDA(cudaFree(d_src));
    LAB_CUDA(cudaSetDevice(dev_dst));
    LAB_CUDA(cudaFree(d_dst));
    LAB_CUDA(cudaFreeHost(h_stage));
    return ms;
}

// UVA form: plain cudaMemcpy with DeviceToDevice kind. Same physical transfer
// as cudaMemcpyPeer once peer access is on.
static float bench_uva(int dev_src, int dev_dst, size_t bytes, const char* label) {
    int* d_src = nullptr;
    int* d_dst = nullptr;
    LAB_CUDA(cudaSetDevice(dev_src));
    LAB_CUDA(cudaMalloc(&d_src, bytes));
    LAB_CUDA(cudaMemset(d_src, 0x7E, bytes));
    LAB_CUDA(cudaSetDevice(dev_dst));
    LAB_CUDA(cudaMalloc(&d_dst, bytes));

    lab::enable_all_peers(2);
    LAB_CUDA(cudaSetDevice(dev_dst));
    cudaStream_t s;
    LAB_CUDA(cudaStreamCreate(&s));
    lab::GpuTimer t;
    t.start(s);
    // The pointers carry device identity; the stream supplies the ordering.
    LAB_CUDA(cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, s));
    t.stop(s);
    float ms = t.elapsed_ms();
    LAB_CUDA(cudaStreamDestroy(s));
    lab::print_bandwidth(label, bytes, ms);

    LAB_CUDA(cudaSetDevice(dev_src));
    LAB_CUDA(cudaFree(d_src));
    LAB_CUDA(cudaSetDevice(dev_dst));
    LAB_CUDA(cudaFree(d_dst));
    return ms;
}

int main(int argc, char** argv) {
    int n_gpus = lab::require_gpus(2);

    size_t bytes = 1ull << 26;  // 64 MiB
    if (argc > 1) bytes = static_cast<size_t>(std::atoll(argv[1])) * sizeof(int);

    std::printf("==== lesson 02: peer copy ====\n");
    lab::enable_all_peers(n_gpus);
    lab::print_peer_matrix(n_gpus);
    std::printf("\nbytes = %.2f MiB\n", bytes / (1024.0 * 1024.0));

    lab::banner("naive: GPU0 -> HOST -> GPU1");
    bench_one(0, 1, bytes, /*through_host=*/true, "naive bounce");

    lab::banner("peer: GPU0 -> GPU1 (cudaMemcpyPeer)");
    bench_one(0, 1, bytes, /*through_host=*/false, "peer NVLink");

    lab::banner("UVA: plain cudaMemcpy DeviceToDevice");
    bench_uva(0, 1, bytes, "UVA D2D");

    std::printf("\nlesson 02 done.\n");
    return 0;
}
