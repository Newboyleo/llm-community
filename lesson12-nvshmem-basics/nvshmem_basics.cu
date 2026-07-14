// lesson12-nvshmem-basics/nvshmem_basics.cu
//
// First NVSHMEM program. Each PE has a symmetric buffer. PE 0's kernel writes
// into PE 1's buffer with nvshmem_int_put; PE 1's kernel reads from PE 0 with
// nvshmem_int_get. The host never issues a copy.

#include <cstdio>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "checks.hpp"
#include "print.hpp"

// Each PE fills its symmetric buffer with 1000*pe + i so we can tell PEs apart.
__global__ void init_kernel(int* buf, int n, int pe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 1000 * pe + i;
}

// PE 0 writes a marker into dst_pe's buffer (one-sided).
__global__ void put_kernel(int* buf, int n, int dst_pe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int val = 1000000 + i;
        nvshmem_int_put(&buf[i], &val, 1, dst_pe);
    }
}

// This PE reads src_pe's buffer into its own local buffer.
__global__ void get_kernel(const int* buf, int* out, int n, int src_pe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = nvshmem_int_g(&buf[i], src_pe);  // get into register, store locally
    }
}

static void print_device_by_pe(int pe, int owner_pe, const char* label, const int* data, int n) {
    nvshmem_barrier_all();
    if (pe == owner_pe) {
        lab::print_device(label, data, n, 4);
        std::fflush(stdout);
    }
    nvshmem_barrier_all();
}

int main() {
    nvshmem_init();
    int pe = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    if (pe == 0) {
        std::printf("==== lesson 12: NVSHMEM basics ====\nn_pes = %d\n", npes);
    }
    if (npes < 2) {
        if (pe == 0) {
            std::fprintf(stderr,
                         "lesson 12 needs at least 2 NVSHMEM PEs.\n"
                         "Run with: CUDA_VISIBLE_DEVICES=0,1 "
                         "/usr/bin/nvshmem_12/nvshmrun -np 2 "
                         "./build/lesson12-nvshmem-basics/nvshmem_basics\n");
        }
        nvshmem_finalize();
        return 1;
    }
    int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    LAB_CUDA(cudaSetDevice(local_pe));

    const int n = 1 << 14;  // 16K ints
    int* buf = (int*)nvshmem_malloc(n * sizeof(int));
    int* local = (int*)nvshmem_malloc(n * sizeof(int));
    if (buf == nullptr || local == nullptr) {
        std::fprintf(stderr, "PE %d: nvshmem_malloc failed\n", pe);
        nvshmem_finalize();
        return 1;
    }

    // init each PE's buffer
    init_kernel<<<(n + 255) / 256, 256>>>(buf, n, pe);
    LAB_CUDA_SYNC();
    nvshmem_barrier_all();   // also implies quiet

    print_device_by_pe(pe, 0, "  PE0 buf", buf, n);
    print_device_by_pe(pe, 1, "  PE1 buf", buf, n);

    // PE 0 puts markers into PE 1's buffer
    if (pe == 0 && npes >= 2) {
        put_kernel<<<(n + 255) / 256, 256>>>(buf, n, /*dst_pe=*/1);
        LAB_CUDA(cudaDeviceSynchronize());
        nvshmem_quiet();
        nvshmem_barrier_all();
    } else {
        nvshmem_barrier_all();
    }

    if (pe == 1) {
        lab::print_device("  PE1 buf after put", buf, n, 4);
        std::fflush(stdout);
    }

    // PE 1 gets PE 0's buffer into local
    if (pe == 1 && npes >= 2) {
        get_kernel<<<(n + 255) / 256, 256>>>(buf, local, n, /*src_pe=*/0);
        LAB_CUDA(cudaDeviceSynchronize());
        nvshmem_quiet();
        lab::print_device("  PE1 local after get", local, n, 4);
        std::fflush(stdout);
    }

    nvshmem_free(buf);
    nvshmem_free(local);
    nvshmem_finalize();
    if (pe == 0) std::printf("\nlesson 12 done.\n");
    return 0;
}
