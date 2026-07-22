// lesson17-mini-deepep/mini_deepep.cu
//
// Mini DeepEP dispatch: device-side NVSHMEM puts into symmetric per-channel
// receive buffers, per-channel ready flags (quiet-guarded), multiple channels
// in parallel. This version makes three DeepEP tradeoffs explicit:
//   1. low-latency vs normal mode: separate kernels and different defaults
//   2. optional FP8 transport: quantize on producer, dequantize in expert
//   3. dispatch/expert overlap: separate streams with a simple SM budget
//
// Simplifications vs real DeepEP:
//   - top-1 routing (real DeepEP: top-k)
//   - receive layout is per-source slotted (src writes to recvbuf[src*Tlocal..])
//     rather than prefix-sum-packed; correct and simple, slightly wasteful.
//   - FP8 is an educational E4M3-style codec with one scale per token.
//   - expert compute is a fake FMA loop, used only to demonstrate overlap.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "checks.hpp"
#include "print.hpp"

enum class DispatchMode { LowLatency, Normal };

struct RunConfig {
    DispatchMode mode = DispatchMode::LowLatency;
    bool use_fp8 = false;
    int channels = 0;
    int chunk_tokens = 0;
    int dispatch_sms = 0;
    int expert_sms = 0;
    int expert_iters = 0;
    int poll_sleep = 0;
};

static bool is_integer(const char* s) {
    if (s == nullptr || *s == '\0') return false;
    if (*s == '-' || *s == '+') ++s;
    if (*s == '\0') return false;
    while (*s) {
        if (!std::isdigit(static_cast<unsigned char>(*s))) return false;
        ++s;
    }
    return true;
}

static bool streq(const char* a, const char* b) {
    return std::strcmp(a, b) == 0;
}

static bool is_bool_token(const char* s) {
    return streq(s, "0") || streq(s, "1") || streq(s, "true") || streq(s, "false") ||
           streq(s, "on") || streq(s, "off") || streq(s, "fp8") || streq(s, "no-fp8");
}

static bool parse_bool_token(const char* s) {
    return streq(s, "1") || streq(s, "true") || streq(s, "on") || streq(s, "fp8");
}

static bool parse_mode(const char* s, DispatchMode* mode) {
    if (streq(s, "ll") || streq(s, "low") || streq(s, "low-latency")) {
        *mode = DispatchMode::LowLatency;
        return true;
    }
    if (streq(s, "normal") || streq(s, "throughput") || streq(s, "bw")) {
        *mode = DispatchMode::Normal;
        return true;
    }
    return false;
}

static const char* mode_name(DispatchMode mode) {
    return mode == DispatchMode::LowLatency ? "low-latency" : "normal";
}

static size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

static size_t dispatch_smem_bytes(int D, int chunk_tokens, bool use_fp8) {
    size_t bytes = 0;
    if (use_fp8) {
        bytes += (size_t)chunk_tokens * D * sizeof(unsigned char);
        bytes = align_up(bytes, alignof(float));
        bytes += (size_t)chunk_tokens * sizeof(float);
    } else {
        bytes = align_up(bytes, alignof(float));
        bytes += (size_t)chunk_tokens * D * sizeof(float);
    }
    bytes = align_up(bytes, alignof(int));
    bytes += (size_t)chunk_tokens * sizeof(int);
    return bytes;
}

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [T] [E] [D] [mode|channels] [fp8] [channels] [chunk] "
                 "[dispatch_sms] [expert_sms] [expert_iters]\n"
                 "  mode: ll|low-latency|normal|throughput\n"
                 "  fp8: 0|1|fp8|no-fp8\n"
                 "examples:\n"
                 "  %s 2048 8 256 low-latency fp8\n"
                 "  %s 65536 8 256 normal fp8 8 32 8 96\n",
                 argv0, argv0, argv0);
}

// ---- routing kernels (lesson 15) ------------------------------------------
__global__ void gate_kernel(const float* x, const float* W, float* logits, int T, int E, int D) {
    int t = blockIdx.x; if (t >= T) return;
    int e = threadIdx.x; if (e >= E) return;
    float s = 0.f;
    for (int d = 0; d < D; ++d) s += x[t * D + d] * W[d * E + e];
    logits[t * E + e] = s;
}

__global__ void top1_kernel(const float* logits, int* assign, int T, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int best = 0; float bv = logits[t * E];
    for (int e = 1; e < E; ++e) { float v = logits[t * E + e]; if (v > bv) { bv = v; best = e; } }
    assign[t] = best;
}

__global__ void count_local_kernel(const int* assign, int* count_row, int Tlocal, int n, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Tlocal) return;
    int e = assign[t]; int dst = e / (E / n);
    atomicAdd(&count_row[dst], 1);
}

// ---- tiny E4M3-style FP8 codec --------------------------------------------
__device__ static inline float fp8_e4m3_decode(unsigned char q) {
    if ((q & 0x7f) == 0) return 0.f;
    float sign = (q & 0x80) ? -1.f : 1.f;
    int exp_bits = (q >> 3) & 0x0f;
    int mant = q & 0x07;
    if (exp_bits == 0) {
        return sign * ldexpf((float)mant / 8.f, -6);
    }
    return sign * ldexpf(1.f + (float)mant / 8.f, exp_bits - 7);
}

__device__ static inline unsigned char fp8_e4m3_encode(float x) {
    if (x == 0.f) return 0;
    unsigned char sign = x < 0.f ? 0x80 : 0x00;
    float ax = fabsf(x);

    int exp2 = 0;
    float m = frexpf(ax, &exp2);  // ax = m * 2^exp2, m in [0.5, 1)
    int unbiased = exp2 - 1;

    if (unbiased > 7) return sign | 0x7e;  // max finite-ish value
    if (unbiased < -6) {
        int sub = (int)lrintf(ldexpf(ax, 6) * 8.f);
        if (sub < 0) sub = 0;
        if (sub > 7) sub = 7;
        return sign | (unsigned char)sub;
    }

    float normalized = ldexpf(m, 1);  // [1, 2)
    int mant = (int)lrintf((normalized - 1.f) * 8.f);
    if (mant == 8) {
        mant = 0;
        ++unbiased;
        if (unbiased > 7) return sign | 0x7e;
    }
    int exp_bits = unbiased + 7;
    return sign | (unsigned char)((exp_bits << 3) | mant);
}

__device__ static inline unsigned char* align_shared(unsigned char* p, size_t a) {
    uintptr_t v = reinterpret_cast<uintptr_t>(p);
    v = (v + a - 1) & ~(uintptr_t)(a - 1);
    return reinterpret_cast<unsigned char*>(v);
}

template <int D, bool UseFp8>
__device__ void shared_layout(unsigned char* raw, int chunk_tokens,
                              float*& fbuf, unsigned char*& qbuf,
                              float*& scales, int*& slots) {
    unsigned char* p = raw;
    fbuf = nullptr;
    qbuf = nullptr;
    scales = nullptr;
    if constexpr (UseFp8) {
        qbuf = p;
        p += (size_t)chunk_tokens * D * sizeof(unsigned char);
        p = align_shared(p, alignof(float));
        scales = reinterpret_cast<float*>(p);
        p += (size_t)chunk_tokens * sizeof(float);
    } else {
        p = align_shared(p, alignof(float));
        fbuf = reinterpret_cast<float*>(p);
        p += (size_t)chunk_tokens * D * sizeof(float);
    }
    p = align_shared(p, alignof(int));
    slots = reinterpret_cast<int*>(p);
}

template <int D, bool UseFp8>
__device__ void flush_chunk(float* recvbuf, unsigned char* recvbuf8, float* recv_scale,
                            float* fbuf, unsigned char* qbuf, float* scales,
                            const int* slots, int collected, int dst) {
    for (int b = 0; b < collected; ++b) {
        size_t slot = (size_t)slots[b];
        if constexpr (UseFp8) {
            nvshmem_float_put(&recv_scale[slot], &scales[b], 1, dst);
            nvshmem_putmem(&recvbuf8[slot * D], &qbuf[(size_t)b * D], (size_t)D, dst);
        } else {
            nvshmem_putmem(&recvbuf[slot * D], &fbuf[(size_t)b * D],
                           (size_t)D * sizeof(float), dst);
        }
    }
    nvshmem_quiet();
}

template <int D, bool UseFp8>
__device__ void stage_token(const float* tokens, int t, int e, int collected,
                            float* fbuf, unsigned char* qbuf, float* scales) {
    if constexpr (UseFp8) {
        float max_abs = 1.f;
        for (int d = 0; d < D; ++d) {
            float v = (d == 0) ? (float)(e + 1) : tokens[(size_t)t * D + d];
            max_abs = fmaxf(max_abs, fabsf(v));
        }
        float scale = fmaxf(max_abs / 240.f, 1.0e-8f);
        scales[collected] = scale;
        for (int d = 0; d < D; ++d) {
            float v = (d == 0) ? (float)(e + 1) : tokens[(size_t)t * D + d];
            qbuf[(size_t)collected * D + d] = fp8_e4m3_encode(v / scale);
        }
    } else {
        for (int d = 0; d < D; ++d) {
            fbuf[(size_t)collected * D + d] =
                (d == 0) ? (float)(e + 1) : tokens[(size_t)t * D + d];
        }
    }
}

template <int D, bool UseFp8>
__device__ void dispatch_one_pair(const float* __restrict__ tokens,
                                  const int* __restrict__ assign,
                                  float* __restrict__ recvbuf,
                                  unsigned char* __restrict__ recvbuf8,
                                  float* __restrict__ recv_scale,
                                  int* __restrict__ ready,
                                  int Tlocal, int n, int E, int nch,
                                  int src_pe, int pair, int chunk_tokens,
                                  unsigned char* raw) {
    int ch = pair / n;
    int dst = pair - ch * n;
    int experts_per_gpu = E / n;

    float* fbuf;
    unsigned char* qbuf;
    float* scales;
    int* slots;
    shared_layout<D, UseFp8>(raw, chunk_tokens, fbuf, qbuf, scales, slots);

    int local_idx = 0;
    int collected = 0;
    for (int t = 0; t < Tlocal; ++t) {
        int e = assign[t];
        if (e / experts_per_gpu != dst) continue;
        if (local_idx % nch != ch) { ++local_idx; continue; }

        stage_token<D, UseFp8>(tokens, t, e, collected, fbuf, qbuf, scales);
        slots[collected] = src_pe * Tlocal + local_idx;
        ++collected;
        ++local_idx;

        if (collected == chunk_tokens) {
            flush_chunk<D, UseFp8>(recvbuf, recvbuf8, recv_scale,
                                   fbuf, qbuf, scales, slots, collected, dst);
            collected = 0;
        }
    }
    if (collected > 0) {
        flush_chunk<D, UseFp8>(recvbuf, recvbuf8, recv_scale,
                               fbuf, qbuf, scales, slots, collected, dst);
    }

    int seq = 1;
    nvshmem_int_put(&ready[ch * n + src_pe], &seq, 1, dst);
    nvshmem_quiet();
}

// Low-latency mode: few channels, one-token chunks, tight consumer polling.
template <int D, bool UseFp8>
__global__ void
__launch_bounds__(64, 1)
dispatch_low_latency_kernel(const float* __restrict__ tokens,
                            const int* __restrict__ assign,
                            float* __restrict__ recvbuf,
                            unsigned char* __restrict__ recvbuf8,
                            float* __restrict__ recv_scale,
                            int* __restrict__ ready,
                            int Tlocal, int n, int E, int nch, int src_pe) {
    if (threadIdx.x != 0) return;
    extern __shared__ unsigned char raw[];
    for (int pair = blockIdx.x; pair < nch * n; pair += gridDim.x) {
        dispatch_one_pair<D, UseFp8>(tokens, assign, recvbuf, recvbuf8, recv_scale,
                                     ready, Tlocal, n, E, nch, src_pe, pair, 1, raw);
    }
}

// Normal mode: more channels and bigger chunks to favor link bandwidth.
template <int D, bool UseFp8>
__global__ void
__launch_bounds__(64, 1)
dispatch_normal_kernel(const float* __restrict__ tokens,
                       const int* __restrict__ assign,
                       float* __restrict__ recvbuf,
                       unsigned char* __restrict__ recvbuf8,
                       float* __restrict__ recv_scale,
                       int* __restrict__ ready,
                       int Tlocal, int n, int E, int nch, int src_pe,
                       int chunk_tokens) {
    if (threadIdx.x != 0) return;
    extern __shared__ unsigned char raw[];
    for (int pair = blockIdx.x; pair < nch * n; pair += gridDim.x) {
        dispatch_one_pair<D, UseFp8>(tokens, assign, recvbuf, recvbuf8, recv_scale,
                                     ready, Tlocal, n, E, nch, src_pe, pair,
                                     chunk_tokens, raw);
    }
}

__device__ static inline void wait_ready_flag(int* ready, int idx, int poll_sleep) {
    volatile int* r = reinterpret_cast<volatile int*>(ready);
    while (r[idx] != 1) {
        if (poll_sleep > 0) __nanosleep((unsigned int)poll_sleep);
    }
}

template <int D, bool UseFp8>
__global__ void
__launch_bounds__(128, 1)
expert_wait_compute_kernel(const float* __restrict__ recvbuf,
                           const unsigned char* __restrict__ recvbuf8,
                           const float* __restrict__ recv_scale,
                           float* __restrict__ expert_out,
                           int* __restrict__ ready,
                           int Tlocal, int n, int nch,
                           int poll_sleep, int expert_iters) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    int lanes = gridDim.x * blockDim.x;

    for (int ch = 0; ch < nch; ++ch) {
        if (threadIdx.x == 0) {
            for (int src = 0; src < n; ++src) {
                wait_ready_flag(ready, ch * n + src, poll_sleep);
            }
            __threadfence_system();
        }
        __syncthreads();

        for (int flat = lane; flat < n * Tlocal; flat += lanes) {
            int slot_in_src = flat % Tlocal;
            if (slot_in_src % nch != ch) continue;

            size_t base = (size_t)flat * D;
            float marker;
            if constexpr (UseFp8) {
                marker = fp8_e4m3_decode(recvbuf8[base]) * recv_scale[flat];
            } else {
                marker = recvbuf[base];
            }
            if (marker == 0.f) continue;

            for (int d = 0; d < D; ++d) {
                float v;
                if constexpr (UseFp8) {
                    v = fp8_e4m3_decode(recvbuf8[base + d]) * recv_scale[flat];
                } else {
                    v = recvbuf[base + d];
                }
                float y = v;
                if (d != 0) {
                    for (int it = 0; it < expert_iters; ++it) {
                        y = fmaf(y, 1.00001f, 0.00001f);
                    }
                }
                expert_out[base + d] = y;
            }
        }
        __syncthreads();
    }
}

template <int D, bool UseFp8>
static void launch_dispatch(DispatchMode mode, dim3 grid, size_t smem, cudaStream_t stream,
                            const float* tokens, const int* assign,
                            float* recvbuf, unsigned char* recvbuf8, float* recv_scale,
                            int* ready, int Tlocal, int n, int E, int nch, int pe,
                            int chunk_tokens) {
    if (mode == DispatchMode::LowLatency) {
        dispatch_low_latency_kernel<D, UseFp8><<<grid, 64, smem, stream>>>(
            tokens, assign, recvbuf, recvbuf8, recv_scale, ready, Tlocal, n, E, nch, pe);
    } else {
        dispatch_normal_kernel<D, UseFp8><<<grid, 64, smem, stream>>>(
            tokens, assign, recvbuf, recvbuf8, recv_scale, ready, Tlocal, n, E, nch,
            pe, chunk_tokens);
    }
}

template <int D, bool UseFp8>
static void launch_expert(dim3 grid, cudaStream_t stream,
                          const float* recvbuf, const unsigned char* recvbuf8,
                          const float* recv_scale, float* expert_out, int* ready,
                          int Tlocal, int n, int nch, int poll_sleep,
                          int expert_iters) {
    expert_wait_compute_kernel<D, UseFp8><<<grid, 128, 0, stream>>>(
        recvbuf, recvbuf8, recv_scale, expert_out, ready, Tlocal, n, nch,
        poll_sleep, expert_iters);
}

int main(int argc, char** argv) {
    int T = 2048, E = 8, D = 256;
    int user_channels = 0, user_chunk = 0, user_dispatch_sms = 0, user_expert_sms = 0;
    int user_expert_iters = 0;
    RunConfig cfg;

    nvshmem_init();
    int n = nvshmem_n_pes();
    int pe = nvshmem_my_pe();
    if (n < 2) { if (pe == 0) std::fprintf(stderr, "needs >=2 PEs\n"); nvshmem_finalize(); return 1; }
    LAB_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));

    cudaDeviceProp prop{};
    LAB_CUDA(cudaGetDeviceProperties(&prop, nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));

    if (argc > 1) T = std::atoi(argv[1]);
    if (argc > 2) E = std::atoi(argv[2]);
    if (argc > 3) D = std::atoi(argv[3]);
    int argi = 4;
    if (argc > argi) {
        if (is_integer(argv[argi])) {
            user_channels = std::atoi(argv[argi++]);  // legacy lesson17 argv[4]
        } else if (!parse_mode(argv[argi++], &cfg.mode)) {
            if (pe == 0) print_usage(argv[0]);
            nvshmem_finalize();
            return 1;
        }
    }
    if (argc > argi && is_bool_token(argv[argi])) cfg.use_fp8 = parse_bool_token(argv[argi++]);
    if (argc > argi) user_channels = std::atoi(argv[argi++]);
    if (argc > argi) user_chunk = std::atoi(argv[argi++]);
    if (argc > argi) user_dispatch_sms = std::atoi(argv[argi++]);
    if (argc > argi) user_expert_sms = std::atoi(argv[argi++]);
    if (argc > argi) user_expert_iters = std::atoi(argv[argi++]);

    if (T <= 0 || E <= 0 || D <= 0) {
        if (pe == 0) std::fprintf(stderr, "T, E, and D must all be positive\n");
        nvshmem_finalize();
        return 1;
    }
    if (E > 1024) {
        if (pe == 0)
            std::fprintf(stderr, "E must be <= 1024 because gate_kernel uses one block with E threads\n");
        nvshmem_finalize();
        return 1;
    }
    if (E % n != 0) { if (pe == 0) std::fprintf(stderr, "E must be divisible by n\n"); nvshmem_finalize(); return 1; }
    if (T % n != 0) { if (pe == 0) std::fprintf(stderr, "T must be divisible by n\n"); nvshmem_finalize(); return 1; }
    if (D != 64 && D != 128 && D != 256) {
        if (pe == 0) std::fprintf(stderr, "D must be one of 64, 128, 256\n");
        nvshmem_finalize();
        return 1;
    }

    cfg.channels = (cfg.mode == DispatchMode::LowLatency) ? 2 : 8;
    cfg.chunk_tokens = (cfg.mode == DispatchMode::LowLatency) ? 1 : 32;
    cfg.dispatch_sms = (cfg.mode == DispatchMode::LowLatency)
                           ? std::min(2, prop.multiProcessorCount)
                           : std::max(1, prop.multiProcessorCount / 4);
    cfg.expert_sms = std::max(1, prop.multiProcessorCount - cfg.dispatch_sms);
    cfg.expert_iters = (cfg.mode == DispatchMode::LowLatency) ? 16 : 96;
    cfg.poll_sleep = (cfg.mode == DispatchMode::LowLatency) ? 0 : 64;

    if (user_channels > 0) cfg.channels = user_channels;
    if (user_chunk > 0) cfg.chunk_tokens = user_chunk;
    if (user_dispatch_sms > 0) cfg.dispatch_sms = user_dispatch_sms;
    if (user_expert_sms > 0) cfg.expert_sms = user_expert_sms;
    if (user_expert_iters > 0) cfg.expert_iters = user_expert_iters;
    if (cfg.mode == DispatchMode::LowLatency) cfg.chunk_tokens = 1;

    cfg.channels = std::max(1, cfg.channels);
    cfg.chunk_tokens = std::max(1, cfg.chunk_tokens);
    if (prop.multiProcessorCount > 1) {
        cfg.dispatch_sms = std::max(1, std::min(cfg.dispatch_sms, prop.multiProcessorCount - 1));
        cfg.expert_sms = std::max(1, std::min(cfg.expert_sms,
                                               prop.multiProcessorCount - cfg.dispatch_sms));
    } else {
        cfg.dispatch_sms = 1;
        cfg.expert_sms = 1;
    }

    size_t smem = dispatch_smem_bytes(D, cfg.chunk_tokens, cfg.use_fp8);
    if (smem > (size_t)prop.sharedMemPerBlock) {
        if (pe == 0) {
            std::fprintf(stderr,
                         "chunk=%d requires %zu bytes shared memory per dispatch block, "
                         "but device default limit is %zu bytes\n",
                         cfg.chunk_tokens, smem, (size_t)prop.sharedMemPerBlock);
        }
        nvshmem_finalize();
        return 1;
    }

    int Tlocal = T / n;
    int dispatch_blocks = std::min(cfg.dispatch_sms, cfg.channels * n);

    if (pe == 0) {
        std::printf("==== lesson 17: mini DeepEP dispatch (optimized knobs) ====\n"
                    "n=%d PEs, T=%d, E=%d, D=%d\n"
                    "mode=%s, fp8=%s, channels=%d, chunk_tokens=%d\n"
                    "SM budget: dispatch_blocks=%d, expert_blocks=%d, expert_iters=%d, poll_sleep=%d\n",
                    n, T, E, D, mode_name(cfg.mode), cfg.use_fp8 ? "on" : "off",
                    cfg.channels, cfg.chunk_tokens, dispatch_blocks, cfg.expert_sms,
                    cfg.expert_iters, cfg.poll_sleep);
    }

    // symmetric allocations. recvbuf is sized n*Tlocal*D so every src has a region.
    float* tokens       = (float*)nvshmem_malloc((size_t)Tlocal * D * sizeof(float));
    float* W            = (float*)nvshmem_malloc((size_t)D * E * sizeof(float));
    float* logits       = (float*)nvshmem_malloc((size_t)Tlocal * E * sizeof(float));
    int* assign         = (int*)nvshmem_malloc((size_t)Tlocal * sizeof(int));
    int* count_row      = (int*)nvshmem_malloc(n * sizeof(int));
    float* recvbuf      = (float*)nvshmem_malloc((size_t)n * Tlocal * D * sizeof(float));
    unsigned char* recvbuf8 =
        (unsigned char*)nvshmem_malloc((size_t)n * Tlocal * D * sizeof(unsigned char));
    float* recv_scale   = (float*)nvshmem_malloc((size_t)n * Tlocal * sizeof(float));
    float* expert_out   = (float*)nvshmem_malloc((size_t)n * Tlocal * D * sizeof(float));
    int* ready          = (int*)nvshmem_malloc((size_t)cfg.channels * n * sizeof(int));

    if (!tokens || !W || !logits || !assign || !count_row || !recvbuf ||
        !recvbuf8 || !recv_scale || !expert_out || !ready) {
        if (pe == 0) std::fprintf(stderr, "nvshmem_malloc failed\n");
        nvshmem_finalize();
        return 1;
    }

    {
        std::vector<float> ht((size_t)Tlocal * D);
        for (int i = 0; i < Tlocal * D; ++i) ht[i] = (float)(pe * 1000 + (i & 0xff));
        LAB_CUDA(cudaMemcpy(tokens, ht.data(), ht.size() * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> hw((size_t)D * E);
        for (int i = 0; i < D * E; ++i) hw[i] = (float)((i * 214013 + 2531011) & 0xff) / 256.f - 0.5f;
        LAB_CUDA(cudaMemcpy(W, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemset(count_row, 0, n * sizeof(int)));
        LAB_CUDA(cudaMemset(recvbuf, 0, (size_t)n * Tlocal * D * sizeof(float)));
        LAB_CUDA(cudaMemset(recvbuf8, 0, (size_t)n * Tlocal * D * sizeof(unsigned char)));
        LAB_CUDA(cudaMemset(recv_scale, 0, (size_t)n * Tlocal * sizeof(float)));
        LAB_CUDA(cudaMemset(expert_out, 0, (size_t)n * Tlocal * D * sizeof(float)));
        LAB_CUDA(cudaMemset(ready, 0, (size_t)cfg.channels * n * sizeof(int)));
    }
    nvshmem_barrier_all();

    // routing
    gate_kernel<<<Tlocal, E>>>(tokens, W, logits, Tlocal, E, D);
    top1_kernel<<<(Tlocal + 255) / 256, 256>>>(logits, assign, Tlocal, E);
    count_local_kernel<<<(Tlocal + 255) / 256, 256>>>(assign, count_row, Tlocal, n, E);
    LAB_CUDA_SYNC();
    nvshmem_barrier_all();

    // gather the global count matrix via one-sided gets (naive AllReduce-sum, host-side)
    std::vector<int> global(n * n, 0);
    for (int r = 0; r < n; ++r)
        for (int d = 0; d < n; ++d)
            global[r * n + d] = nvshmem_int_g(&count_row[d], r);

    if (pe == 0) {
        std::printf("\nglobal count[src][dst]:\n       ");
        for (int d = 0; d < n; ++d) std::printf("dst%-4d", d);
        std::printf("\n");
        for (int s = 0; s < n; ++s) {
            std::printf("src%-3d ", s);
            for (int d = 0; d < n; ++d) std::printf("%-7d", global[s * n + d]);
            std::printf("\n");
        }
    }

    cudaStream_t dispatch_stream, expert_stream;
    LAB_CUDA(cudaStreamCreate(&dispatch_stream));
    LAB_CUDA(cudaStreamCreate(&expert_stream));

    // Start the expert-side stream first; it polls ready flags while dispatch runs.
    dim3 expert_grid(cfg.expert_sms);
    if (cfg.use_fp8) {
        if (D == 256)      launch_expert<256, true>(expert_grid, expert_stream, recvbuf, recvbuf8, recv_scale, expert_out, ready, Tlocal, n, cfg.channels, cfg.poll_sleep, cfg.expert_iters);
        else if (D == 128) launch_expert<128, true>(expert_grid, expert_stream, recvbuf, recvbuf8, recv_scale, expert_out, ready, Tlocal, n, cfg.channels, cfg.poll_sleep, cfg.expert_iters);
        else               launch_expert<64, true>(expert_grid, expert_stream, recvbuf, recvbuf8, recv_scale, expert_out, ready, Tlocal, n, cfg.channels, cfg.poll_sleep, cfg.expert_iters);
    } else {
        if (D == 256)      launch_expert<256, false>(expert_grid, expert_stream, recvbuf, recvbuf8, recv_scale, expert_out, ready, Tlocal, n, cfg.channels, cfg.poll_sleep, cfg.expert_iters);
        else if (D == 128) launch_expert<128, false>(expert_grid, expert_stream, recvbuf, recvbuf8, recv_scale, expert_out, ready, Tlocal, n, cfg.channels, cfg.poll_sleep, cfg.expert_iters);
        else               launch_expert<64, false>(expert_grid, expert_stream, recvbuf, recvbuf8, recv_scale, expert_out, ready, Tlocal, n, cfg.channels, cfg.poll_sleep, cfg.expert_iters);
    }

    dim3 dispatch_grid(dispatch_blocks);
    if (cfg.use_fp8) {
        if (D == 256)      launch_dispatch<256, true>(cfg.mode, dispatch_grid, smem, dispatch_stream, tokens, assign, recvbuf, recvbuf8, recv_scale, ready, Tlocal, n, E, cfg.channels, pe, cfg.chunk_tokens);
        else if (D == 128) launch_dispatch<128, true>(cfg.mode, dispatch_grid, smem, dispatch_stream, tokens, assign, recvbuf, recvbuf8, recv_scale, ready, Tlocal, n, E, cfg.channels, pe, cfg.chunk_tokens);
        else               launch_dispatch<64, true>(cfg.mode, dispatch_grid, smem, dispatch_stream, tokens, assign, recvbuf, recvbuf8, recv_scale, ready, Tlocal, n, E, cfg.channels, pe, cfg.chunk_tokens);
    } else {
        if (D == 256)      launch_dispatch<256, false>(cfg.mode, dispatch_grid, smem, dispatch_stream, tokens, assign, recvbuf, recvbuf8, recv_scale, ready, Tlocal, n, E, cfg.channels, pe, cfg.chunk_tokens);
        else if (D == 128) launch_dispatch<128, false>(cfg.mode, dispatch_grid, smem, dispatch_stream, tokens, assign, recvbuf, recvbuf8, recv_scale, ready, Tlocal, n, E, cfg.channels, pe, cfg.chunk_tokens);
        else               launch_dispatch<64, false>(cfg.mode, dispatch_grid, smem, dispatch_stream, tokens, assign, recvbuf, recvbuf8, recv_scale, ready, Tlocal, n, E, cfg.channels, pe, cfg.chunk_tokens);
    }

    LAB_CUDA(cudaStreamSynchronize(dispatch_stream));
    LAB_CUDA(cudaStreamSynchronize(expert_stream));
    nvshmem_barrier_all();

    // verify: every token processed by this PE's fake expert has expert id mapping here
    int experts_per_gpu = E / n;
    int total_incoming = 0;
    for (int s = 0; s < n; ++s) total_incoming += global[s * n + pe];
    std::vector<float> host((size_t)n * Tlocal * D);
    LAB_CUDA(cudaMemcpy(host.data(), expert_out, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    int seen = 0; bool ok = true;
    for (int s = 0; s < n && ok; ++s) {
        for (int i = 0; i < Tlocal; ++i) {
            float e_f = host[((size_t)s * Tlocal + i) * D];
            if (e_f == 0.f) continue;  // empty slot (no token from src s here)
            int e = (int)(e_f + 0.5f) - 1;
            if (e / experts_per_gpu != pe) { ok = false; break; }
            ++seen;
        }
    }
    if (seen != total_incoming) ok = false;
    std::printf("PE%d: %d tokens arrived, all map here after expert overlap: %s\n",
                pe, seen, ok ? "YES" : "NO");

    LAB_CUDA(cudaStreamDestroy(dispatch_stream));
    LAB_CUDA(cudaStreamDestroy(expert_stream));
    nvshmem_free(tokens); nvshmem_free(W); nvshmem_free(logits); nvshmem_free(assign);
    nvshmem_free(count_row); nvshmem_free(recvbuf); nvshmem_free(recvbuf8);
    nvshmem_free(recv_scale); nvshmem_free(expert_out); nvshmem_free(ready);
    nvshmem_finalize();
    if (pe == 0) std::printf("\nlesson 17 done.\n");
    return 0;
}
