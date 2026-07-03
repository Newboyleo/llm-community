# Lesson 11 — Events

> Events are the scalpel for cross-stream dependencies. Where
> `cudaDeviceSynchronize` is a sledgehammer (waits for everything), an event
> lets stream B wait for *one specific* earlier operation on stream A. This is
> how you build correct, overlapped pipelines.

---

# Overview

## What are we building?

Two demonstrations:

1. **Precise dependency wiring.** Stream A produces data; stream B must wait
   for *that* data before consuming it. We record an event after A's producer
   and make B wait on it. No global sync, no over-synchronization.
2. **GPU-side timing.** The cleanest way to time a piece of GPU work is to
   record events before and after it *on the same stream* and read
   `cudaEventElapsedTime`. (We've used `lab::GpuTimer` for this since lesson 1;
   here we look under the hood.)

```
Without events (over-sync):            With events (precise):
  stream A: produce                     stream A: produce ─ recEvt
  cudaDeviceSynchronize()  <- waits ALL  stream B:        waitEvt ─ consume
  stream B: consume
```

## Why does it matter?

Every overlapped pipeline in production inference is a graph of streams wired
by events. NCCL's internal scheduler, DeepEP's multi-channel dispatch, the
copy/compute overlap in vLLM/SGLang — all are event graphs. If you only know
`cudaDeviceSynchronize`, you can only write *serial* GPU code; events are the
gateway to *concurrent* GPU code.

## Where is it used in LLM inference?

- **DeepEP dispatch:** "expert `e`'s compute stream waits on the event
  recording the arrival of `e`'s tokens on the dispatch stream." Per-expert,
  per-channel — not a global barrier.
- **Layer pipeline:** "layer N+1's matmul stream waits on layer N's
  AllReduce-complete event."
- **CUDA Graph capture:** events are how the graph records dependencies
  between captured nodes.

---

# Goal

- Record events, make streams wait on them, and measure elapsed time.
- Build a 3-stage overlapped pipeline (copy → compute → copy) wired *only* by
  events — no `cudaDeviceSynchronize` in the steady state.
- Understand why `cudaStreamWaitEvent` is non-blocking on the issuing CPU
  thread (it enqueues a wait, doesn't stall).

---

# Background

## The event lifecycle

```c
cudaEvent_t e;
cudaEventCreate(&e);          // (or CreateWithFlags, pool variant)
cudaEventRecord(e, streamA);  // "mark here on streamA"
cudaStreamWaitEvent(streamB, e, 0);  // "streamB waits until e is reached"
cudaEventSynchronize(e);      // (host blocks until e reached — rare in hot paths)
cudaEventDestroy(e);
```

`cudaEventRecord` and `cudaStreamWaitEvent` are **asynchronous** — they enqueue
work; the CPU doesn't stall. The wait happens on the GPU, in stream order.

## Two flag flavors

- `cudaEventDefault` — supports timing (`cudaEventElapsedTime`) and sync.
- `cudaEventDisableTiming` — faster to record; use when you only need the
  dependency, not the timing. NCCL/DeepEP use this for the bulk of their
  event traffic.

## The ping-pong pipeline

The classic pattern: two buffers, two streams. While stream A computes on
buffer 0, stream B copies buffer 1 in (or out). Events mark "buffer N ready"
so the consumer stream waits only on the right buffer.

---

# Architecture Diagram

```
   3-stage pipeline, two streams, events = vertical bars:

   stream copy :  [copy0]──evt0──[copy1]──evt1──[copy2]──evt2──
                          │              │              │
   stream compute:       wait0──[kern0]──wait1──[kern1]──wait2──[kern2]

   No cudaDeviceSynchronize in steady state — only event-based handoffs.
```

---

# Source Code Walkthrough

`events.cu`:

- Reuses `busy_kernel` from lesson 10.
- `demo_dependency()` — stream A writes a buffer and records `evtA`; stream B
  waits on `evtA` then reads the buffer. Show that removing the wait breaks
  correctness (race).
- `demo_timing()` — record events around a kernel on one stream, read
  `cudaEventElapsedTime`. This is exactly `lab::GpuTimer`.
- `demo_pipeline()` — two-stream copy/compute ping-pong over 3 chunks, wired
  entirely by events.

Key shape (dependency):

```c
cudaEvent_t ready;
cudaEventCreate(&ready);
producer<<<..., streamA>>>(...);
cudaEventRecord(ready, streamA);
cudaStreamWaitEvent(streamB, ready, 0);   // B waits for producer
consumer<<<..., streamB>>>(...);
```

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target events
```

---

# Run

```bash
./build/lesson11-events/events
```

---

# Expected Output

```
==== lesson 11: events ====

[dependency] stream B waits on stream A's event
  with wait:    consumer saw producer's data: YES
  without wait: consumer saw producer's data: NO  (race!)

[timing] event-based GPU timer
  kernel elapsed: 40.2 ms

[pipeline] 3-stage copy/compute, event-wired
  serial baseline:   126.3 ms   (3 × (copy + kernel))
  event-overlapped:   84.7 ms   (copy hidden behind kernel)
```

---

# Experiment

1. **Remove the wait.** In `demo_dependency`, comment out
   `cudaStreamWaitEvent`. Run a few times — sometimes correct, sometimes not.
   The race is real. This is the #1 bug in hand-rolled stream code.
2. **Disable timing.** Replace `cudaEventCreate` with
   `cudaEventCreateWithFlags(&e, cudaEventDisableTiming)`. Record/sync cost
   drops. Use this for hot-path dependency events.
3. **Too many events.** Create 100 000 events in a loop. They're cheap but not
   free; measure the host overhead. Production code reuses a small pool.
4. **nsys timeline.** Capture `nsys profile` of the pipeline demo. You'll see
   the streams' lanes and the event arrows between them — the visual
   confirmation of overlap.

---

# Performance Analysis

- **Event record/wait** are ~1–5 µs of host overhead each. Cheap, but in a
  tight per-token loop they add up — batch them.
- **`cudaStreamWaitEvent` does not stall the CPU.** It enqueues a wait on the
  GPU. The CPU keeps enqueueing. The only CPU stall is
  `cudaEventSynchronize` / `cudaStreamSynchronize` at the end.
- **The pipeline wins exactly when stage time > event overhead.** For
  microsecond-scale stages (tiny tokens), event overhead dominates and the
  pipeline loses. DeepEP uses **batched** events (one per batch of tokens) to
  push stage time well above the event floor.

---

# Exercises

1. **Generalize to N stages.** Parameterize the pipeline over arbitrary
   chunk count and stream count. Build the event dependency graph
   programmatically.
2. **Event pool.** Replace create/destroy per stage with a fixed pool of
   recycled events. Measure the host-overhead reduction.
3. **Capture as a CUDA Graph.** Wrap the pipeline in `cudaStreamBeginCapture`
   / `EndCapture` and instantiate a graph. The event overhead disappears
   (the graph replays without host involvement). This is how production
   inference engines hit sub-microsecond per-layer overhead.

---

# DeepEP Connection

```
Lesson 11  events (precise cross-stream deps)
   ↓
DeepEP     per-channel, per-expert event graph:
             dispatch_stream  ->  records "tokens for expert e arrived"
             expert_stream_e  ->  waits on that event, runs expert kernel
             combine_stream   ->  waits on all expert_stream_e events
           No global barrier anywhere — exactly the lesson-11 pattern,
           multiplied across channels and experts.
```

When you read DeepEP's `dispatch`/`combine` host orchestration in lesson 18,
the structure is: *for each channel, for each expert, record/wait events*.
Today's two-stream demo is the two-node version of that graph.
