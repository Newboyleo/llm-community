// lesson09-ring-buffer/ring_buffer.cu
//
// A single-producer / single-consumer ring buffer on one GPU. The producer
// block writes values; the consumer block reads them. Coordination is via
// head/tail counters with __threadfence() ordering. This is the data structure
// DeepEP's channel buffers are built from (across GPUs, via NVSHMEM, lesson 13).

#include <cstdio>
#include <vector>

#include "checks.hpp"
#include "print.hpp"
#include "timing.hpp"

struct Ring {
    int* slots;            // device buffer of `cap` ints
    volatile int* head;    // producer-only writer
    volatile int* tail;    // consumer-only writer
    int cap;
};

// Push one value. Spins while full. The __threadfence() BEFORE head advance
// guarantees the consumer (which reads head then reads the slot) sees the data.
__device__ void ring_push(Ring r, int value) {
    for (;;) {
        int h = *r.head;
        int t = *r.tail;
        if (h - t < r.cap) {
            r.slots[h % r.cap] = value;
            __threadfence();          // publish data before head
            *r.head = h + 1;
            return;
        }
        // full — spin
    }
}

// Pop one value. Spins while empty.
__device__ int ring_pop(Ring r) {
    for (;;) {
        int h = *r.head;
        int t = *r.tail;
        if (h != t) {
            __threadfence();          // ensure we see producer's data
            int v = r.slots[t % r.cap];
            __threadfence();          // publish consumption before tail
            *r.tail = t + 1;
            return v;
        }
        // empty — spin
    }
}

__global__ void producer_kernel(Ring r, int count) {
    // single-thread producer
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < count; ++i) ring_push(r, i);
}

__global__ void consumer_kernel(Ring r, int* out, int count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < count; ++i) out[i] = ring_pop(r);
}

int main(int argc, char** argv) {
    int cap = 8;
    int count = 100000;
    if (argc > 1) cap = std::atoi(argv[1]);
    if (argc > 2) count = std::atoi(argv[2]);

    std::printf("==== lesson 09: ring buffer (SPSC, single GPU) ====\n");
    std::printf("capacity = %d, count = %d\n", cap, count);

    int* d_slots;  LAB_CUDA(cudaMalloc(&d_slots, cap * sizeof(int)));
    int* d_head;   LAB_CUDA(cudaMalloc(&d_head, sizeof(int)));
    int* d_tail;   LAB_CUDA(cudaMalloc(&d_tail, sizeof(int)));
    int* d_out;    LAB_CUDA(cudaMalloc(&d_out, count * sizeof(int)));
    LAB_CUDA(cudaMemset(d_head, 0, sizeof(int)));
    LAB_CUDA(cudaMemset(d_tail, 0, sizeof(int)));

    Ring r{d_slots, (volatile int*)d_head, (volatile int*)d_tail, cap};

    cudaStream_t sp, sc;
    LAB_CUDA(cudaStreamCreate(&sp));
    LAB_CUDA(cudaStreamCreate(&sc));

    lab::GpuTimer t;
    t.start(sp);
    producer_kernel<<<1, 1, 0, sp>>>(r, count);
    consumer_kernel<<<1, 1, 0, sc>>>(r, d_out, count);
    LAB_CUDA(cudaStreamSynchronize(sp));
    LAB_CUDA(cudaStreamSynchronize(sc));
    t.stop(sp);

    std::vector<int> out(count);
    LAB_CUDA(cudaMemcpy(out.data(), d_out, count * sizeof(int), cudaMemcpyDeviceToHost));

    bool in_order = true;
    for (int i = 0; i < count; ++i) if (out[i] != i) { in_order = false; break; }
    std::printf("producer pushed %d values\n", count);
    std::printf("consumer popped  %d values\n", count);
    lab::print_host("first 8 popped", out.data(), 8, 8);
    std::printf("all values in order: %s\n", in_order ? "YES" : "NO");
    std::printf("total time %.2f ms\n", t.elapsed_ms());

    LAB_CUDA(cudaFree(d_slots));
    LAB_CUDA(cudaFree(d_head));
    LAB_CUDA(cudaFree(d_tail));
    LAB_CUDA(cudaFree(d_out));
    LAB_CUDA(cudaStreamDestroy(sp));
    LAB_CUDA(cudaStreamDestroy(sc));
    std::printf("\nlesson 09 done.\n");
    return 0;
}
