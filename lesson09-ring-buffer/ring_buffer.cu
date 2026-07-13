// lesson09-ring-buffer/ring_buffer.cu
//
// A single-producer / single-consumer ring buffer on one GPU. The producer
// block writes values; the consumer block reads them. Coordination is via
// head/tail counters with __threadfence() ordering. This is the data structure
// DeepEP's channel buffers are built from (across GPUs, via NVSHMEM, lesson 13).

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
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

static void log_host(const char* msg) {
    std::printf("[host] %s\n", msg);
    std::fflush(stdout);
}

static bool query_stream_done(cudaStream_t stream, bool* done) {
    cudaError_t e = cudaStreamQuery(stream);
    if (e == cudaSuccess) {
        *done = true;
        return true;
    }
    if (e == cudaErrorNotReady) {
        *done = false;
        return true;
    }
    LAB_CUDA(e);
    return false;
}

static bool wait_streams_with_logs(cudaStream_t producer_stream,
                                   cudaStream_t consumer_stream,
                                   int timeout_sec) {
    auto begin = std::chrono::steady_clock::now();
    int last_printed_sec = -1;

    for (;;) {
        bool producer_done = false;
        bool consumer_done = false;
        query_stream_done(producer_stream, &producer_done);
        query_stream_done(consumer_stream, &consumer_done);

        auto now = std::chrono::steady_clock::now();
        int elapsed_sec = static_cast<int>(
            std::chrono::duration_cast<std::chrono::seconds>(now - begin).count());

        if (elapsed_sec != last_printed_sec) {
            std::printf("[host] waiting %2ds: producer=%s, consumer=%s\n",
                        elapsed_sec,
                        producer_done ? "done" : "running",
                        consumer_done ? "done" : "running");
            std::fflush(stdout);
            last_printed_sec = elapsed_sec;
        }

        if (producer_done && consumer_done) return true;

        if (elapsed_sec >= timeout_sec) {
            std::printf("[host] timeout after %ds. If producer is still running, "
                        "it likely filled the ring and is spinning before the "
                        "consumer kernel made progress.\n",
                        timeout_sec);
            std::fflush(stdout);
            cudaDeviceReset();
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

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
    printf("[device producer] start count=%d cap=%d\n", count, r.cap);
    for (int i = 0; i < count; ++i) ring_push(r, i);
    printf("[device producer] done\n");
}

__global__ void consumer_kernel(Ring r, int* out, int count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    printf("[device consumer] start count=%d cap=%d\n", count, r.cap);
    for (int i = 0; i < count; ++i) out[i] = ring_pop(r);
    printf("[device consumer] done\n");
}

int main(int argc, char** argv) {
    int cap = 8;
    int count = 100;
    if (argc > 1) cap = std::atoi(argv[1]);
    if (argc > 2) count = std::atoi(argv[2]);

    std::printf("==== lesson 09: ring buffer (SPSC, single GPU) ====\n");
    std::printf("capacity = %d, count = %d\n", cap, count);
    std::fflush(stdout);

    log_host("allocating device buffers");
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
    log_host("created producer and consumer streams");

    lab::GpuTimer t;
    t.start(sp);
    log_host("launching producer kernel on producer stream");
    producer_kernel<<<1, 1, 0, sp>>>(r, count);
    LAB_CUDA(cudaGetLastError());
    log_host("launching consumer kernel on consumer stream");
    consumer_kernel<<<1, 1, 0, sc>>>(r, d_out, count);
    LAB_CUDA(cudaGetLastError());

    if (!wait_streams_with_logs(sp, sc, 10)) {
        std::fprintf(stderr, "[host] aborting after timeout; CUDA context was reset.\n");
        return 2;
    }

    t.stop(sp);
    LAB_CUDA(cudaStreamSynchronize(sp));
    LAB_CUDA(cudaStreamSynchronize(sc));
    log_host("both streams completed");

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
