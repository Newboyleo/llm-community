# Lesson 10 — CUDA Streams

> Streams are how you get concurrency out of a GPU. Two independent kernels on
> two streams can run simultaneously; a copy and a kernel on different streams
> can overlap. This lesson makes overlap visible — and shows why every
> performance lesson from here on relies on it.

---

# Overview

## What are we building?

Three small experiments that make stream behavior *observable*:

1. **Serial baseline:** two kernels on the default stream — they run
   back-to-back, total time = sum.
2. **Concurrent kernels:** the same two kernels on **two streams** — they run
   in parallel, total time ≈ max (if the GPU has spare SMs).
3. **Copy/compute overlap:** an H2D copy on stream A overlapped with a kernel
   on stream B. The classic "hide transfer behind compute" pattern.

```
Serial (one stream):    ████ kernel A ████ ████ kernel B ████     t = A + B
Concurrent (two streams): ████ kernel A ████
                          ████ kernel B ████                       t ≈ max(A,B)
Copy/compute overlap:    ████ H2D  ████
                            ████ kernel ████                       t ≈ max(copy, kernel)
```

## Why does it matter?

Every multi-GPU collective in this course issues many transfers and wants them
*in flight at once*. Streams are the mechanism. NCCL runs each "channel" on its
own stream; DeepEP runs each dispatch channel on its own stream; the host
orchestrates them with events (lesson 11). If you can't reason about stream
overlap, you can't reason about any of those systems' performance.

## Where is it used in LLM inference?

- **Overlapping AllReduce with the next layer's compute** (the bread and
  butter of DP training throughput).
- **Overlapping MoE dispatch with expert compute** (DeepEP's whole pitch: start
  expert `e`'s compute as soon as its tokens arrive, don't wait for all
  experts' tokens).
- **H2D weight prefetching** while the GPU computes the current layer.

---

# Goal

- Create streams, issue work on them, and synchronize correctly.
- *See* overlap in a timeline (we use `cudaDeviceSynchronize` + timing; in real
  life you'd use Nsight Systems).
- Understand the **default-stream serialization** gotcha and why
  `--default-stream per-thread` exists.

---

# Background

## Stream ordering

Work in the same stream executes **in order**. Work in different streams is
**unordered relative to each other** — the runtime may run them concurrently if
resources allow. "Resources allow" is the catch: 2 kernels that each need 100%
of SMs won't actually overlap, even on separate streams.

## Synchronization

- `cudaStreamSynchronize(s)` — wait for all work on `s`.
- `cudaDeviceSynchronize()` — wait for all work on all streams (heavy; use
  sparingly in hot paths).
- `cudaEventRecord / cudaStreamWaitEvent` — express *only* the dependencies
  you need (lesson 11). This is the scalpel; the two above are sledgehammers.

## The default-stream gotcha

The **legacy default stream** (stream 0) synchronizes with *all* other streams
on the current thread. So if you mix `kernel<<<>>>` (default stream) with work
on a created stream, they serialize. Compile with
`-default-stream per-thread` to give each thread its own non-synchronizing
default. NCCL and NVSHMEM assume per-thread default; we set it in CMake.

---

# Architecture Diagram

```
   Two-stream overlap test:

   stream A:  ──▶ [ kernel A (40 ms) ] ──▶
   stream B:  ──▶ [ kernel B (40 ms) ] ──▶
   wall clock:                  ≈ 40-50 ms (overlap)   vs   80 ms (serial)
```

---

# Source Code Walkthrough

`streams.cu`:

- `__global__ void busy_kernel(int ms)` — a kernel that spins long enough to
  take ~`ms` milliseconds. Uses `clock64()` to time itself from inside.
- `run_serial()` — launch two busy kernels on stream 0, time with `GpuTimer`.
- `run_concurrent()` — launch one on `streamA`, one on `streamB`, time.
- `run_copy_overlap()` — issue an H2D copy on `streamA` and a busy kernel on
  `streamB`; measure wall time vs the sum.

Key shape:

```c
cudaStream_t a, b;
cudaStreamCreate(&a); cudaStreamCreate(&b);
busy_kernel<<<1,1,0,a>>>(40);   // ~40ms on stream a
busy_kernel<<<1,1,0,b>>>(40);   // ~40ms on stream b
cudaStreamSynchronize(a);
cudaStreamSynchronize(b);
// wall time ≈ 40-50ms, not 80ms  => they overlapped
```

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target streams
```

---

# Run

```bash
./build/lesson10-streams/streams
```

---

# Expected Output

```
==== lesson 10: CUDA streams ====

[serial] two 40ms kernels on default stream
  time 81.2 ms   (≈ 2×40 — no overlap)

[concurrent] two 40ms kernels on two streams
  time 44.3 ms   (≈ 40 — overlapped! GPU ran both at once)

[copy/compute overlap] 64MiB H2D + 40ms kernel on separate streams
  H2D alone       2.3 ms
  kernel alone   40.1 ms
  overlapped     40.4 ms   (copy hid behind kernel)
```

---

# Experiment

1. **Starve the SMs.** Launch two kernels that each use *all* SMs (e.g.,
   `<<<N, 1024>>>` where N is the SM count). Now they *can't* overlap —
   confirm the concurrent time is back to ~80ms. Overlap requires spare
   resources.
2. **Three, four, eight streams.** How many kernels can you overlap? You'll
   hit a limit (SMs, then scheduler slots). This is why "more streams = more
   concurrency" stops scaling.
3. **The default-stream trap.** Without `-default-stream per-thread`, launch
   one kernel on a created stream and one on the default. Confirm they
   serialize. Then enable the flag and watch them overlap.
4. **Bad sync.** Replace `cudaStreamSynchronize` with `cudaDeviceSynchronize`
   in a hot loop. Profile the overhead — each `cudaDeviceSynchronize` is a
   system call and can cost ~10 µs.

---

# Performance Analysis

- **Overlap is free only when resources permit.** Two small kernels overlap
  fully; two huge ones don't. Always confirm overlap with a profiler
  (Nsight Systems), never assume.
- **Copy/compute overlap** requires the copy to be on a **non-default stream**
  *and* smaller than the compute it hides. If the copy is bigger, the kernel
  finishes first and waits — no benefit.
- **The async APIs (`cudaMemcpyAsync`, kernel launches) return immediately.**
  The CPU is free to enqueue more work. The skill is *not* in making calls
  async — they already are — but in *not* synchronizing until you must. Every
  `cudaDeviceSynchronize` is a lost overlap opportunity.

---

# Exercises

1. **Pipeline three stages.** Streams A, B, C doing copy→compute→copy on
   consecutive chunks. Issue chunk 0 fully, then chunk 1's copy overlaps with
   chunk 0's compute, etc. This is the canonical double-buffering pipeline.
2. **Measure launch overhead.** Launch 10 000 no-op kernels on one stream.
   Divide total time by 10 000. You'll get ~3–10 µs/launch — the cost every
   lesson so far has been paying per `cudaMemcpyPeerAsync`.
3. **CUDA Graphs.** Capture the three-stage pipeline into a `cudaGraph` and
   replay it. Launch overhead drops to ~1 µs. This is how production inference
   engines keep per-layer overhead low.

---

# DeepEP Connection

```
Lesson 10  CUDA streams (concurrency, overlap)
   ↓
Lesson 11  events (precise cross-stream dependencies)
   ↓
DeepEP     N dispatch channels = N streams, each issuing ring-buffer writes;
           the host wires them with events so channel k's tokens land before
           expert compute on channel k starts.
```

DeepEP's **multi-channel** dispatch is *literally* "run N copies of lesson 9's
ring buffer on N streams." The streams give you the concurrency; events (next
lesson) give you the dependencies. Today is the foundation.
