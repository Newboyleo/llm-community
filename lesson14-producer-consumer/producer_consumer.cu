// lesson14-producer-consumer/producer_consumer.cu
//
// Cross-GPU producer/consumer with explicit ready/done flag handshake.
// Producer puts a batch, quiets, sets ready=seq; consumer polls ready==seq,
// processes, sets done=seq; producer polls done==seq then reuses the slot.

#include <cstdio>
#include <vector>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "checks.hpp"
#include "print.hpp"

struct Channel {
    int* slots;   // symmetric, cap * batch_size ints
    int* ready;   // symmetric, producer writes (via put), consumer reads
    int* done;    // symmetric, consumer writes (via put), producer reads
    int cap;
    int batch_size;
};

// Producer on PE0 sends `n_batches` batches to PE1.
__global__ void producer_kernel(Channel ch, int n_batches, int cons_pe) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int batch_ints = ch.batch_size;
    size_t bb = batch_ints * sizeof(int);
    extern __shared__ int smem[];
    int* scratch = smem;  // single-thread producer; small batch fits in shared mem

    for (int b = 0; b < n_batches; ++b) {
        int slot = b % ch.cap;
        int seq = b + 1;  // 1-based; 0 means empty
        // fill scratch with b's data: batch_index * batch_size + i
        for (int i = 0; i < batch_ints; ++i) scratch[i] = b * batch_ints + i;
        nvshmem_putmem(ch.slots + slot * batch_ints, scratch, bb, cons_pe);
        nvshmem_quiet();
        nvshmem_int_put(ch.ready + slot, &seq, 1, cons_pe);
        nvshmem_quiet();
        // wait for done
        int d;
        do { d = nvshmem_int_g(ch.done + slot, cons_pe); } while (d != seq);
    }
}

// Consumer on PE1 receives `n_batches` batches.
__global__ void consumer_kernel(Channel ch, int n_batches, int* out, int prod_pe) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int batch_ints = ch.batch_size;
    for (int b = 0; b < n_batches; ++b) {
        int slot = b % ch.cap;
        int seq = b + 1;
        int r;
        do { r = ch.ready[slot]; } while (r != seq);   // poll local ready
        // copy slot into out (process)
        for (int i = 0; i < batch_ints; ++i)
            out[b * batch_ints + i] = ch.slots[slot * batch_ints + i];
        // ack
        nvshmem_int_put(ch.done + slot, &seq, 1, prod_pe);
        nvshmem_quiet();
    }
}

int main(int argc, char** argv) {
    int n_batches = 100, batch_size = 256, cap = 8;
    if (argc > 1) n_batches = std::atoi(argv[1]);
    if (argc > 2) batch_size = std::atoi(argv[2]);
    if (argc > 3) cap = std::atoi(argv[3]);

    nvshmem_init();
    int pe = nvshmem_my_pe(), npes = nvshmem_n_pes();
    if (npes < 2) {
        if (pe == 0) std::fprintf(stderr, "needs >=2 PEs\n");
        nvshmem_finalize(); return 1;
    }
    if (pe == 0)
        std::printf("==== lesson 14: producer/consumer (signaled) ====\n"
                    "batches=%d, batch_size=%d, capacity=%d\n",
                    n_batches, batch_size, cap);

    Channel ch;
    ch.cap = cap; ch.batch_size = batch_size;
    ch.slots = (int*)nvshmem_malloc((size_t)cap * batch_size * sizeof(int));
    ch.ready = (int*)nvshmem_malloc(cap * sizeof(int));
    ch.done  = (int*)nvshmem_malloc(cap * sizeof(int));
    LAB_CUDA(cudaMemset(ch.ready, 0, cap * sizeof(int)));
    LAB_CUDA(cudaMemset(ch.done, 0, cap * sizeof(int)));
    nvshmem_barrier_all();

    int* out = (int*)nvshmem_malloc((size_t)n_batches * batch_size * sizeof(int));

    cudaStream_t s; LAB_CUDA(cudaStreamCreate(&s));
    size_t smem = batch_size * sizeof(int);
    if (pe == 0)
        producer_kernel<<<1, 1, smem, s>>>(ch, n_batches, 1);
    else if (pe == 1)
        consumer_kernel<<<1, 1, 0, s>>>(ch, n_batches, out, 0);
    LAB_CUDA(cudaStreamSynchronize(s));
    nvshmem_barrier_all();

    if (pe == 1) {
        std::vector<int> h((size_t)n_batches * batch_size);
        LAB_CUDA(cudaMemcpy(h.data(), out, h.size() * sizeof(int), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int b = 0; b < n_batches && ok; ++b)
            for (int i = 0; i < batch_size; ++i)
                if (h[b * batch_size + i] != b * batch_size + i) { ok = false; break; }
        std::printf("PE1 consumed %d batches\n", n_batches);
        lab::print_host("batch 0 first 4", h.data(), 4, 4);
        lab::print_host("batch N first 4", h.data() + (n_batches - 1) * batch_size, 4, 4);
        std::printf("all batches correct: %s\n", ok ? "YES" : "NO");
    }

    LAB_CUDA(cudaStreamDestroy(s));
    nvshmem_free(ch.slots); nvshmem_free(ch.ready); nvshmem_free(ch.done);
    nvshmem_free(out);
    nvshmem_finalize();
    if (pe == 0) std::printf("\nlesson 14 done.\n");
    return 0;
}
