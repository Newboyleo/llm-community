// lesson13-nvshmem-ring-buffer/nvshmem_ring_buffer.cu
//
// Lesson 9's SPSC ring buffer, but across GPUs via NVSHMEM. PE0 (producer)
// pushes values that land in PE1's (consumer) symmetric slots. Coordination
// is head/tail in symmetric memory + nvshmem_quiet before head advances.

#include <cstdio>
#include <vector>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "checks.hpp"
#include "print.hpp"

struct SymmRing {
    int* slots;   // symmetric, cap ints
    int* head;    // symmetric, owned by producer (writes), read by consumer
    int* tail;    // symmetric, owned by consumer (writes), read by producer
    int cap;
};

// Producer on PE `prod` pushes `count` values to consumer PE `cons`.
__global__ void producer_kernel(SymmRing r, int count, int cons_pe) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int head_local = 0;
    for (int i = 0; i < count; ++i) {
        // wait while full: head_local - remote_tail == cap
        int tail_remote;
        do {
            tail_remote = nvshmem_int_g(r.tail, cons_pe);
        } while (head_local - tail_remote >= r.cap);

        int slot = head_local % r.cap;
        int val = i;
        nvshmem_int_put(&r.slots[slot], &val, 1, cons_pe);  // data -> consumer
        nvshmem_quiet();                                    // data landed
        ++head_local;
        nvshmem_int_put(r.head, &head_local, 1, cons_pe);   // head -> consumer
        nvshmem_quiet();                                    // head landed
    }
}

// Consumer on PE `cons` pops `count` values from producer PE `prod`.
__global__ void consumer_kernel(SymmRing r, int* out, int count, int prod_pe) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int tail_local = 0;
    for (int i = 0; i < count; ++i) {
        int head_seen;
        do {
            head_seen = *r.head;   // producer put head here; local read
        } while (head_seen == tail_local);  // empty -> spin

        int slot = tail_local % r.cap;
        out[i] = r.slots[slot];   // local read of symmetric slot
        ++tail_local;
        // echo tail back to producer so it can see "not full"
        nvshmem_int_put(r.tail, &tail_local, 1, prod_pe);
        nvshmem_quiet();
    }
}

int main(int argc, char** argv) {
    int cap = 8, count = 10000;
    if (argc > 1) cap = std::atoi(argv[1]);
    if (argc > 2) count = std::atoi(argv[2]);

    nvshmem_init();
    int pe = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (pe == 0) std::fprintf(stderr, "needs >=2 PEs (CUDA_VISIBLE_DEVICES=0,1)\n");
        nvshmem_finalize();
        return 1;
    }
    if (pe == 0)
        std::printf("==== lesson 13: NVSHMEM ring buffer ====\n"
                    "PE0=producer, PE1=consumer, capacity=%d, count=%d\n", cap, count);

    int* slots = (int*)nvshmem_malloc(cap * sizeof(int));
    int* head  = (int*)nvshmem_malloc(sizeof(int));
    int* tail  = (int*)nvshmem_malloc(sizeof(int));
    int* out   = (int*)nvshmem_malloc(count * sizeof(int));
    LAB_CUDA(cudaMemset(head, 0, sizeof(int)));
    LAB_CUDA(cudaMemset(tail, 0, sizeof(int)));
    nvshmem_barrier_all();

    SymmRing r{slots, head, tail, cap};

    cudaStream_t s;
    LAB_CUDA(cudaStreamCreate(&s));
    if (pe == 0) {
        producer_kernel<<<1, 1, 0, s>>>(r, count, /*cons_pe=*/1);
    } else if (pe == 1) {
        consumer_kernel<<<1, 1, 0, s>>>(r, out, count, /*prod_pe=*/0);
    }
    LAB_CUDA(cudaStreamSynchronize(s));
    nvshmem_barrier_all();

    if (pe == 1) {
        std::vector<int> h(count);
        LAB_CUDA(cudaMemcpy(h.data(), out, count * sizeof(int), cudaMemcpyDeviceToHost));
        bool in_order = true;
        for (int i = 0; i < count; ++i) if (h[i] != i) { in_order = false; break; }
        std::printf("PE1 popped %d values\n", count);
        lab::print_host("first 8", h.data(), 8, 8);
        std::printf("in order: %s\n", in_order ? "YES" : "NO");
    }

    LAB_CUDA(cudaStreamDestroy(s));
    nvshmem_free(slots); nvshmem_free(head); nvshmem_free(tail); nvshmem_free(out);
    nvshmem_finalize();
    if (pe == 0) std::printf("\nlesson 13 done.\n");
    return 0;
}
