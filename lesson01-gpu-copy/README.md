# Lesson 01 — GPU Copy

> The first program in the lab. One GPU. One buffer. One `cudaMemcpy`.
> Everything else in the course is a generalization of this.

---

# Overview

## What are we building?

A program that allocates an array on the **host** (CPU), copies it to the
**device** (GPU), doubles every element with a kernel, and copies it back.

```
   HOST                         DEVICE
┌──────────┐   cudaMemcpy      ┌──────────┐
│  h_x[]   │  ───────────────▶ │  d_x[]   │
│ 1,2,3,4  │   H2D             │          │
└──────────┘                   └────┬─────┘
                                    │ kernel: d_x[i] *= 2
                                    ▼
┌──────────┐   cudaMemcpy      ┌──────────┐
│  h_y[]   │  ◀─────────────── │  d_x[]   │
│ 2,4,6,8  │   D2H             │ 2,4,6,8  │
└──────────┘                   └──────────┘
```

## Why does it matter?

This is the **only** lesson with a single GPU, so it is your chance to lock in
three things before we add a second GPU:

1. The four CUDA memory spaces you can copy between
   (`Host→Device`, `Device→Host`, `Device→Device`, `Host→Host`).
2. That **the copy is synchronous** — `cudaMemcpy` blocks the CPU until done.
3. That `cudaMalloc` gives you a **device pointer**, which you must never
   dereference on the host.

Get these three right and the rest of the course is just "more GPUs and more
clever about *when* you copy."

## Where is it used in LLM inference?

Everywhere you see a weight or a KV-cache tensor that lives on the GPU but was
loaded from CPU RAM. The H2D copy of weights at model load, the D2H copy of
logits when sampling on CPU, the device-to-device copy when reshuffling a KV
cache — all are `cudaMemcpy` under the hood.

---

# Goal

After this lesson you should be able to:

- name the four `cudaMemcpyKind` directions and when each fires,
- explain why `cudaMemcpy` is the wrong tool for measuring *GPU memory bandwidth*
  (it crosses the PCIe bus, not the HBM),
- read a `GpuTimer` measurement and convert ms → GB/s,
- say what `cudaMalloc` / `cudaFree` cost and why we hoist them out of hot loops.

---

# Background

## The four copy directions

```c
enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,  // plain memcpy on the CPU
    cudaMemcpyHostToDevice   = 1,  // over PCIe/NVLink, host RAM → GPU HBM
    cudaMemcpyDeviceToHost   = 2,  // the reverse
    cudaMemcpyDeviceToDevice = 3,  // GPU → GPU (same device here)
};
```

`cudaMemcpy(dst, src, bytes, kind)` is **synchronous**: the host thread blocks
until the transfer completes (and, for D2H, until the result is visible in host
memory). Lesson 10 introduces the async version.

## Bandwidth you should expect

| Direction        | Typical link        | Bandwidth          |
|------------------|---------------------|--------------------|
| H2D / D2H        | PCIe Gen4 x16       | ~25 GB/s           |
| H2D / D2H        | PCIe Gen5 x16       | ~50 GB/s           |
| D2D (same GPU)   | HBM (A100: ~2 TB/s) | ~1500–1900 GB/s    |
| D2D (peer GPU)   | NVLink              | 100–500 GB/s       | ← lesson 02

So a 1 GiB H2D copy on a Gen4 box should take ~43 ms. If you see ~1 ms you are
measuring something else (probably the kernel, not the copy).

## `cudaMalloc` is expensive

`cudaMalloc` may grab and pin a chunk of the GPU's virtual address space; it is
**not** free. In tight loops we allocate once outside the loop and reuse. The
same is true for `cudaEventCreate` (lesson 11) and `cudaStreamCreate` (lesson 10).

---

# Architecture Diagram

```
                     one GPU
   ┌─────────────────────────────────────────────┐
   │  HOST (CPU page-able memory)                │
   │   h_x  ──┐                                   │
   │          │  cudaMemcpy H2D                   │
   │          ▼                                   │
   │  DEVICE (GPU HBM)                           │
   │   d_x  ──► kernel double<<<>>> ──►  d_x*2    │
   │          │                                   │
   │          │  cudaMemcpy D2H                   │
   │          ▼                                   │
   │  HOST   h_y  (verify)                       │
   └─────────────────────────────────────────────┘
```

---

# Source Code Walkthrough

`gpu_copy.cu`:

- `init_host(h, n)` — fills the host buffer with `1,2,3,...`.
- `__global__ void double_inplace(T* x, int n)` — the kernel. One thread per
  element, `x[i] *= 2`. Generic on `T` so we can reuse it for `int`/`float`.
- `main()` — does the H2D → kernel → D2H round trip **and** measures the H2D
  and D2H legs separately with `lab::GpuTimer`, then prints bandwidth.

Key lines:

```c
// device pointer — do NOT dereference on the host
int* d_x;
LAB_CUDA(cudaMalloc(&d_x, n * sizeof(int)));

// H2D: synchronous, blocks the CPU
LAB_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(int), cudaMemcpyHostToDevice));

// launch: 256 threads/block, enough blocks to cover n
double_inplace<int><<<blocks, 256>>>(d_x, n);
LAB_CUDA_SYNC();   // catches launch + async kernel errors

// D2H
LAB_CUDA(cudaMemcpy(h_y.data(), d_x, n * sizeof(int), cudaMemcpyDeviceToHost));
```

`LAB_CUDA` and `LAB_CUDA_SYNC` come from `common/checks.hpp`. They turn a
`cudaError_t` into a clear `file:line` abort — nothing more.

---

# Build

```bash
# from the repo root
cmake -S . -B build && cmake --build build -j --target gpu_copy
```

Or, from inside this lesson:

```bash
cmake -S . -B build && cmake --build build -j
./build/gpu_copy
```

---

# Run

```bash
./build/lesson01-gpu-copy/gpu_copy
# optionally pick a size (default 1<<24 ints = 64 MiB)
./build/lesson01-gpu-copy/gpu_copy 16777216
```

---

# Expected Output

```
==== lesson 01: GPU copy ====
device 0: NVIDIA H100  (compute 90)
n = 16777216 ints (64.00 MiB)

H2D  64.00 MiB ... done
D2H  64.00 MiB ... done

==== correctness ====
h_x[0..4] = [1, 2, 3, 4]
h_y[0..4] = [2, 4, 6, 8]
OK

==== timing (round trip, no kernel) ====
H2D  64.00 MiB  2.31 ms   27803.9 GB/s  (25900.0 GiB/s)   ← WRONG, see below
...
```

Wait — that H2D number is absurd. What's going on?

`cudaMemcpy` of host memory that is **pageable** (ordinary `malloc`/`vector`)
is copied by the driver through a **staging buffer**: the driver first copies
the pages into pinned memory, then DMAs that. For small sizes the staging copy
dominates and the timer is really measuring the host memcpy, not the PCIe DMA.

To measure the *real* PCIe bandwidth we need **pinned** host memory
(`cudaMallocHost`). The program does both and prints them side by side so you
can see the difference — that difference *is* the lesson.

---

# Experiment

1. **Pageable vs pinned.** The program runs both. Divide the pinned D2H time
   into the byte count. Is it close to the ~25 GB/s (Gen4) / ~50 GB/s (Gen5)
   number from the Background table?
2. **Size sweep.** Run with `1<<16`, `1<<20`, `1<<24`, `1<<26`. Notice the
   pinned bandwidth *climbs* with size — small transfers are latency-bound,
   large ones are bandwidth-bound. Find the crossover.
3. **`cudaMalloc` in the hot loop.** Move the `cudaMalloc`/`cudaFree` *inside*
   the timed loop. How much does the round-trip time blow up? (Answer: a lot.
   This is why every later lesson allocates once.)
4. **Default-stream gotcha.** The kernel and the copies are all on the default
   stream, so they serialize. We'll exploit stream concurrency in lesson 10.

---

# Performance Analysis

Three regimes, three different bottlenecks:

| Size        | Bottleneck               | What you see                  |
|-------------|--------------------------|-------------------------------|
| < 4 KiB     | launch latency (~5 µs)   | time ~constant                |
| 4 KiB–1 MiB | PCIe packet overhead     | bandwidth climbs linearly     |
| > 16 MiB    | PCIe bandwidth           | flat at ~25 GB/s (pinned)     |

The **pageable** line never reaches the pinned ceiling because of the extra
host-side staging copy — that copy runs at host-memory bandwidth and is the
hidden tax on `std::vector`-backed transfers. Pinning your host buffers is the
single cheapest bandwidth win in CUDA.

> **Why this matters for LLM inference:** model weights are loaded once at
> startup (H2D, batched, pinned, asynchronous — lesson 10) and then never
> move. The H2D bandwidth *during inference* is therefore near zero; the
> interesting bandwidth is GPU↔GPU (lessons 2+) and GPU↔HBM (kernel-side).

---

# Exercises

1. **Make it async.** Replace `cudaMemcpy` with `cudaMemcpyAsync` on a stream
   you create, and `cudaStreamSynchronize` before reading `h_y`. Confirm the
   result is identical. (We do this properly in lesson 10.)
2. **Measure D2D.** Allocate two device buffers and time
   `cudaMemcpy(..., cudaMemcpyDeviceToDevice)`. This runs at HBM bandwidth
   (~1.5–2 TB/s on an H100), far above PCIe. *This is the ceiling every
   multi-GPU lesson is trying to approach.*
3. **Break it on purpose.** Dereference `d_x` on the host (`printf("%d", d_x[0])`).
   Observe the segfault. This is why device pointers are dangerous and why
   unified memory (lesson 12) is appealing.

---

# DeepEP Connection

DeepEP's whole job is to move tensors **between GPUs** with minimum latency.
Today's lesson is the trivial case (one GPU, host round-trip); lessons 2–17
strip away the host and the "one GPU" assumption one at a time:

```
Lesson 01  cudaMemcpy H2D/D2H            (host in the loop)
Lesson 02  cudaMemcpyPeer D2D            (host out of the loop, NVLink in)
Lesson 08  AllToAll                      (the MoE dispatch skeleton)
Lesson 17  mini-DeepEP dispatch          (low-latency, channel-parallel)
   ↓
DeepEP    internode_dispatch / intranode_dispatch
```

The mental move from "copy a buffer" (today) to "dispatch a token to the right
expert on the right GPU" (lesson 17) is just adding:
- **a destination rank** (which GPU),
- **a destination offset** (which expert's buffer slot),
- **overlap** (many transfers in flight at once).

That's it. Hold onto that.
