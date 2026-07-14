// lesson11-events/events.cu
//
// Events: precise cross-stream dependencies + GPU-side timing. We build a
// 3-stage copy/compute pipeline wired entirely by events (no device sync in
// steady state).

#include <cstdio>
#include <vector>

#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

#include "checks.hpp"
#include "timing.hpp"

namespace {

struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

void nvtx_mark_chunk(const char* phase, int i) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s chunk %d", phase, i);
    nvtxRangePushA(name);
}

}  // namespace

__global__ void busy_kernel(float target_ms) {
    unsigned long long start = clock64();
    unsigned long long need = (unsigned long long)(target_ms * 1e6);
    while (clock64() - start < need) { /* spin */ }
}

// Producer writes a flag; consumer reads it. Correct only if consumer waits
// on the producer's event.
__global__ void write_flag(int* flag, int val) { *flag = val; }

static void demo_dependency() {
    std::printf("[dependency] stream B waits on stream A's event\n");
    int* d_flag;
    LAB_CUDA(cudaMalloc(&d_flag, sizeof(int)));
    cudaStream_t sa, sb;
    LAB_CUDA(cudaStreamCreate(&sa));
    LAB_CUDA(cudaStreamCreate(&sb));

    // WITH wait: correct
    {
        LAB_CUDA(cudaMemsetAsync(d_flag, 0, sizeof(int), sa));
        cudaEvent_t ready;
        LAB_CUDA(cudaEventCreate(&ready));
        write_flag<<<1, 1, 0, sa>>>(d_flag, 42);
        LAB_CUDA(cudaEventRecord(ready, sa));
        LAB_CUDA(cudaStreamWaitEvent(sb, ready, 0));   // sb waits for sa
        int* d_out;
        LAB_CUDA(cudaMalloc(&d_out, sizeof(int)));
        // copy flag -> out on sb (only valid after sa wrote it)
        LAB_CUDA(cudaMemcpyAsync(d_out, d_flag, sizeof(int), cudaMemcpyDeviceToDevice, sb));
        LAB_CUDA(cudaStreamSynchronize(sb));
        int host = 0;
        LAB_CUDA(cudaMemcpy(&host, d_out, sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("  with wait:    consumer saw %d (expected 42): %s\n",
                    host, host == 42 ? "YES" : "NO");
        LAB_CUDA(cudaEventDestroy(ready));
        LAB_CUDA(cudaFree(d_out));
    }
    // WITHOUT wait: race — may or may not see 42
    {
        LAB_CUDA(cudaMemsetAsync(d_flag, 0, sizeof(int), sa));
        write_flag<<<1, 1, 0, sa>>>(d_flag, 7);
        // deliberately no event/wait
        int* d_out;
        LAB_CUDA(cudaMalloc(&d_out, sizeof(int)));
        LAB_CUDA(cudaMemcpyAsync(d_out, d_flag, sizeof(int), cudaMemcpyDeviceToDevice, sb));
        LAB_CUDA(cudaStreamSynchronize(sb));
        int host = 0;
        LAB_CUDA(cudaMemcpy(&host, d_out, sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("  without wait: consumer saw %d (expected 7): %s  <- race\n",
                    host, host == 7 ? "YES" : "NO");
        LAB_CUDA(cudaFree(d_out));
    }
    LAB_CUDA(cudaStreamDestroy(sa));
    LAB_CUDA(cudaStreamDestroy(sb));
    LAB_CUDA(cudaFree(d_flag));
    std::printf("\n");
}

static void demo_timing() {
    std::printf("[timing] event-based GPU timer\n");
    cudaEvent_t a, b;
    LAB_CUDA(cudaEventCreate(&a));
    LAB_CUDA(cudaEventCreate(&b));
    LAB_CUDA(cudaEventRecord(a));
    busy_kernel<<<1, 1>>>(40.0f);
    LAB_CUDA(cudaEventRecord(b));
    LAB_CUDA(cudaEventSynchronize(b));
    float ms = 0;
    LAB_CUDA(cudaEventElapsedTime(&ms, a, b));
    std::printf("  kernel elapsed: %.1f ms\n", ms);
    LAB_CUDA(cudaEventDestroy(a));
    LAB_CUDA(cudaEventDestroy(b));
    std::printf("\n");
}

// 3-stage copy/compute pipeline. stream_copy does H2D for chunk i; stream_comp
// computes on chunk i. stream_comp waits (via event) on stream_copy for chunk i.
static void demo_pipeline() {
    NvtxRange pipeline_range("demo_pipeline");
    std::printf("[pipeline] 3-stage copy/compute, event-wired\n");
    const int N = 3;
    size_t bytes = 1ull << 28;  // 256 MiB per chunk
    std::vector<int*> d(N);
    int* h;
    cudaStream_t scopy, scomp;
    {
        NvtxRange setup_range("pipeline setup");
        LAB_CUDA(cudaMallocHost(&h, bytes));
        for (int i = 0; i < N; ++i) LAB_CUDA(cudaMalloc(&d[i], bytes));

        LAB_CUDA(cudaStreamCreate(&scopy));
        LAB_CUDA(cudaStreamCreate(&scomp));
        nvtxNameCudaStreamA(scopy, "pipeline-copy");
        nvtxNameCudaStreamA(scomp, "pipeline-compute");
    }

    // serial baseline
    lab::GpuTimer t;
    float serial_ms = 0;
    {
        NvtxRange serial_range("serial baseline");
        t.start(scopy);
        for (int i = 0; i < N; ++i) {
            nvtx_mark_chunk("serial H2D", i);
            LAB_CUDA(cudaMemcpyAsync(d[i], h, bytes, cudaMemcpyHostToDevice, scopy));
            LAB_CUDA(cudaStreamSynchronize(scopy));
            nvtxRangePop();

            nvtx_mark_chunk("serial compute", i);
            busy_kernel<<<1, 1, 0, scomp>>>(40.0f);
            LAB_CUDA(cudaStreamSynchronize(scomp));
            nvtxRangePop();
        }
        t.stop(scopy);
        serial_ms = t.elapsed_ms();
    }

    // event-overlapped: copy i overlaps with compute i-1
    std::vector<cudaEvent_t> ready(N);
    for (int i = 0; i < N; ++i) LAB_CUDA(cudaEventCreate(&ready[i]));
    float overlap_ms = 0;
    {
        NvtxRange overlap_range("event-overlapped");
        t.start(scopy);
        for (int i = 0; i < N; ++i) {
            nvtx_mark_chunk("overlap enqueue", i);
            LAB_CUDA(cudaMemcpyAsync(d[i], h, bytes, cudaMemcpyHostToDevice, scopy));
            LAB_CUDA(cudaEventRecord(ready[i], scopy));           // chunk i copied
            LAB_CUDA(cudaStreamWaitEvent(scomp, ready[i], 0));    // compute waits for it
            busy_kernel<<<1, 1, 0, scomp>>>(40.0f);
            nvtxRangePop();
        }
        LAB_CUDA(cudaStreamSynchronize(scopy));
        LAB_CUDA(cudaStreamSynchronize(scomp));
        t.stop(scopy);
        overlap_ms = t.elapsed_ms();
    }

    std::printf("  serial baseline:   %.1f ms\n", serial_ms);
    std::printf("  event-overlapped:  %.1f ms   (copy hidden behind compute)\n",
                overlap_ms);
    std::printf("  nsys: nsys profile --trace=cuda,nvtx -o events_pipeline "
                "./build/lesson11-events/events\n");
    std::printf("        nsys stats --report nvtx_pushpop_sum,cuda_gpu_trace "
                "events_pipeline.nsys-rep\n");

    {
        NvtxRange cleanup_range("pipeline cleanup");
        for (int i = 0; i < N; ++i) {
            LAB_CUDA(cudaEventDestroy(ready[i]));
            LAB_CUDA(cudaFree(d[i]));
        }
        LAB_CUDA(cudaFreeHost(h));
        LAB_CUDA(cudaStreamDestroy(scopy));
        LAB_CUDA(cudaStreamDestroy(scomp));
    }
}

int main() {
    std::printf("==== lesson 11: events ====\n\n");
    demo_pipeline();
    std::printf("\nlesson 11 done.\n");
    return 0;
}
