#pragma once
// multigpu.hpp — tiny helpers for the multi-GPU lessons (2-11).
//
// Two responsibilities:
//   1. count GPUs and sanity-check we have at least `want`.
//   2. enable bidirectional peer access between every pair (when supported)
//      so cudaMemcpyPeer goes over NVLink instead of bouncing through host.
//
// We deliberately do NOT wrap cudaMemcpyPeer itself — the lessons call it
// directly so the reader sees the real API.

#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "checks.hpp"

namespace lab {

inline int gpu_count() {
    int n = 0;
    LAB_CUDA(cudaGetDeviceCount(&n));
    return n;
}

// Make sure we have at least `want` GPUs; otherwise abort with a clear message.
inline int require_gpus(int want) {
    int n = gpu_count();
    if (n < want) {
        lab::fail(__FILE__, __LINE__,
                  "this lesson needs " + std::to_string(want) +
                  " GPUs but found " + std::to_string(n) +
                  ". See docs/HARDWARE.md.");
    }
    return n;
}

// Enable peer access for every ordered pair (i,j) that supports it. Prints a
// small table so the reader can see which links are NVLink vs PCIe.
inline void enable_all_peers(int n) {
    std::printf("[peers] device topology (%d GPUs):\n", n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int can = 0;
            LAB_CUDA(cudaDeviceCanAccessPeer(i, j, &can));
            if (can) {
                // Enabling is idempotent; ignore "already enabled" errors.
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    LAB_CUDA(e);
                }
            }
        }
    }
}

// Human-readable link label between i and j using the cudaDeviceProp
// pciBusId comparison (a stand-in; real topology comes from nvidia-smi topo).
inline void print_peer_matrix(int n) {
    std::printf("        ");
    for (int j = 0; j < n; ++j) std::printf("GPU%-4d", j);
    std::printf("\n");
    for (int i = 0; i < n; ++i) {
        std::printf("GPU%-4d ", i);
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                std::printf("  X    ");
            } else {
                int can = 0;
                cudaDeviceCanAccessPeer(i, j, &can);
                std::printf("%-7s", can ? "P2P" : "sys");
            }
        }
        std::printf("\n");
    }
    std::printf("  (run `nvidia-smi topo -m` to see NVLink vs PCIe)\n");
}

}  // namespace lab
