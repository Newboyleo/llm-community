# Lesson 09 — Ring Buffer

> A fixed-size circular buffer that lets a **producer** and **consumer** share
> data without locks. This is the data structure DeepEP (and NVSHMEM, and
> NCCL) uses to stream tokens between GPUs with minimum latency.

---

# Overview

## What are we building?

A **single-producer, single-consumer (SPSC) ring buffer** that lives in GPU
memory. The producer writes entries into slots; the consumer reads them out.
The two never block each other because:

- the producer only ever writes the **head**,
- the consumer only ever writes the **tail**,
- each reads the other's counter to know how much room / data is available.

We build it *within a single GPU first* (so you can see the logic without P2P
noise), then explain how the *exact same* structure spans two GPUs in lesson
13 (NVSHMEM ring buffer).

```
   slots:  [ 0 ][ 1 ][ 2 ][ 3 ][ 0 ][ 1 ]...   (capacity = 4)
              ^                                   ^
            tail (consumer)                     head (producer)
```

## Why does it matter?

A ring buffer is the canonical **lock-free** communication structure. Every
low-latency GPU communication system — NCCL's channels, NVSHMEM examples,
DeepEP's dispatch buffers — is, at its core, a ring buffer (or a bank of them)
with a producer on one GPU and a consumer on another. Build this once, on one
GPU, and the multi-GPU version (lesson 13) is just "the head/tail live in
symmetric memory."

## Where is it used in LLM inference?

- **DeepEP's channel buffers:** each dispatch channel is a ring buffer the
  producer GPU writes tokens into and the consumer GPU reads expert-input
  tokens out of.
- **KV-cache streaming** between pipeline stages.
- **NCCL's "channel"** is internally a ring of buffers with producer/consumer
  semantics per step.

---

# Goal

- Implement an SPSC ring buffer with `head`/`tail` counters and a `__threadfence`
  (or `volatile` + fence) ordering discipline.
- Understand the **happens-before** contract: data written *before* the head
  advances must be visible *after* the consumer reads the new head.
- See why the buffer is never quite full (one slot reserved) — the standard
  ambiguity-avoidance trick.

---

# Background

## The two-counter protocol

```
head : written by PRODUCER only. "I have filled slots [tail, head)."
tail : written by CONSUMER only. "I have consumed slots [tail, head)."
```

- Producer: `while (head - tail == capacity) /* full, wait */; write slot head%cap; fence; head++`.
- Consumer: `while (head == tail) /* empty, wait */; read slot tail%cap; fence; tail++`.

The **fence** between the data write and the head advance is the whole game.
Without it, the GPU may reorder the head store *before* the data stores, and
the consumer reads garbage. With `__threadfence()`, the producer guarantees
all prior stores are globally visible before the head increment is.

## Full vs empty

- `head == tail` → empty.
- `head - tail == capacity` → full.

So we can store `capacity` items, but the "one slot reserved" variant (full
when `head - tail == capacity - 1`) avoids any modular-arithmetic ambiguity.
We use the counters-with-capacity form here; it's what DeepEP uses.

---

# Architecture Diagram

```
   SPSC ring buffer (capacity = 8), single GPU.

   slots:  [ D ][ D ][ . ][ . ][ . ][ . ][ . ][ . ]
             ↑                   ↑
            tail=2              head=2
            (consumer has       (producer has
             read 2)             written 2)

   Producer thread/block:                Consumer thread/block:
     write slot[head]                      while (head==tail) spin;
     __threadfence();                      __threadfence();  // see new head
     head++;                               read slot[tail];
                                            __threadfence();
                                            tail++;
```

In our single-GPU demo, producer and consumer run as **two blocks on the same
GPU**. In lesson 13 they run on **different GPUs** over NVSHMEM.

---

# Source Code Walkthrough

`ring_buffer.cu`:

- `struct Ring { int* slots; volatile int* head; volatile int* tail; int cap; }` —
  the buffer state. `head`/`tail` are `volatile` so the compiler doesn't
  hoist the loads out of the spin loops.
- `__device__ void ring_push(Ring r, int value)` — spin while full; write;
  `__threadfence()`; advance head.
- `__device__ int ring_pop(Ring r)` — spin while empty; `__threadfence()`
  (to see the producer's data); read; `__threadfence()`; advance tail.
- `producer_kernel` pushes `M` values; `consumer_kernel` pops `M` values and
  writes them to an output array. Launched on separate streams; the consumer
  blocks until data appears.

Key lines (producer):

```c
while (atomicAdd((int*)r.head, 0) - *r.tail == r.cap) ;  // full -> spin
r.slots[*r.head % r.cap] = value;
__threadfence();        // <-- data visible BEFORE head advances
atomicAdd((int*)r.head, 1);
```

(We use `atomicAdd(...,0)` as a safe volatile read; on modern CUDA you can
also use `__ldcg` or just `volatile` loads.)

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target ring_buffer
```

---

# Run

```bash
./build/lesson09-ring-buffer/ring_buffer
./build/lesson09-ring-buffer/ring_buffer 8 100000   # capacity, count
```

---

# Expected Output

```
==== lesson 09: ring buffer (SPSC, single GPU) ====
capacity = 8, count = 100000

producer pushed 100000 values
consumer popped  100000 values
first 8 popped: [0, 1, 2, 3, 4, 5, 6, 7]
all values in order: YES
total time 3.21 ms
```

The "all values in order: YES" line is the correctness proof: despite the
producer and consumer running concurrently with only `head`/`tail` for
coordination, every value arrives in order.

---

# Experiment

1. **Remove the fence.** Delete the `__threadfence()` before `head++`. On
   many GPUs the output will still be correct (the bug is a data race that
   only triggers under reorderings). On some it will corrupt. This is the
   scariest kind of bug — usually works, occasionally wrong. *This is why
   fences exist.*
2. **Capacity = 1.** The buffer degenerates to strict ping-pong: producer
   writes, consumer reads, producer writes, … Throughput drops, latency per
   item rises. This is the regime DeepEP's *low-latency* dispatch lives in.
3. **Capacity = count.** Effectively a queue with no backpressure. Highest
   throughput, most memory. Find the sweet spot for your GPU.
4. **Two consumers.** Now it's MPSC. The simple `tail++` breaks — two
   consumers can read the same slot. You need an atomic `tail++` *and* to
   remember which slot you claimed. (DeepEP uses one consumer per channel to
   stay SPSC.)

---

# Performance Analysis

- **Throughput** is bounded by the slower of producer/consumer, plus the fence
  cost (~10s of ns on modern GPUs, but it forces a memory-system flush).
- **Latency per item** ≈ fence cost + one HBM round-trip for the slot. For
  capacity=1, throughput ≈ `1 / latency`. For large capacity, the producer
  runs ahead and throughput is limited only by HBM write bandwidth.
- **The fence is the tax.** Every push pays one `__threadfence`. DeepEP
  amortizes this by pushing **batches** (many tokens per head advance) — one
  fence per batch, not per token. That single optimization is worth ~10× on
  small-token workloads.

---

# Exercises

1. **Batch the fence.** Change `ring_push` to push `B` values per head
   advance (head counts *batches*, not items). Measure the throughput gain at
   B=1,4,16,64. This is the DeepEP batch trick.
2. **Make it MPMC.** Use `atomicAdd(head, 1)` to claim a producer slot and
   `atomicAdd(tail, 1)` to claim a consumer slot. Confirm correctness with 4
   producers and 4 consumers.
3. **Move the buffer to unified memory.** Allocate with `cudaMallocManaged`
   and run producer on GPU 0, consumer on GPU 1. It works (slowly) — and is a
  stepping stone to lesson 13's NVSHMEM version.

---

# DeepEP Connection

```
Lesson 09  SPSC ring buffer (single GPU, __threadfence)
   ↓
Lesson 13  NVSHMEM ring buffer (same struct, head/tail in symmetric memory)
   ↓
Lesson 14  producer/consumer across GPUs (signaled completion)
   ↓
DeepEP     ChannelBuffer: per-channel ring buffer, producer = dispatching GPU,
           consumer = expert GPU. Batching + multiple channels hide the fence
           and link latency.
```

DeepEP's `ChannelBuffer` is literally this lesson's ring buffer, with:

- the slots in **NVSHMEM symmetric memory** (lesson 12) so the producer's
  writes land directly in the consumer GPU's HBM,
- the head/tail counters also symmetric, polled by the consumer,
- **batched head advances** (one fence per batch of tokens),
- **N channels in parallel** to hide per-channel latency.

When you read DeepEP's `dispatch_buffer`/`combine_buffer` code in lesson 18,
recognize it as today's `Ring` wearing a suit.
