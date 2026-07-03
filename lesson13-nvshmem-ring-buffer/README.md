# Lesson 13 — NVSHMEM Ring Buffer

> Lesson 9's SPSC ring buffer, reborn across GPUs. The head lives on the
> producer PE, the tail on the consumer PE, and the slots in symmetric memory.
> A kernel on the producer streams tokens straight into the consumer's HBM.

---

# Overview

## What are we building?

An SPSC ring buffer where **producer and consumer are on different GPUs**.
Producer PE 0 pushes values that land in PE 1's symmetric buffer; consumer PE
1 pops them. The only coordination is the head/tail counters — also in
symmetric memory — and an `nvshmem_quiet` before the head advances.

```
   PE 0 (producer)                          PE 1 (consumer)
   ┌──────────────┐  nvshmem_int_put        ┌──────────────┐
   │ push loop:   │ ──────────────────────▶ │ slots[]      │
   │  write slot  │   (data, into PE1 HBM)  │ (symmetric)  │
   │  quiet       │                         │              │
   │  head++      │ ──────────────────────▶ │ head (symm)  │
   └──────────────┘   (head, into PE1)      │              │
                                          pop loop:        │
                                            while(head==tail) spin
                                            read slot
                                            tail++
```

## Why does it matter?

This is the **direct ancestor of DeepEP's channel buffer**. The producer is
the dispatching GPU sending tokens; the consumer is the expert GPU receiving
them. Once you can stream values GPU→GPU with nothing but head/tail counters
for coordination, you have the machinery for low-latency dispatch — you just
need (a) batches instead of scalars, (b) N channels in parallel, and (c) a
router deciding which channel each token takes. Those are lessons 14–17.

## Where is it used in LLM inference?

- **DeepEP `ChannelBuffer`** — exactly this structure, batched and multiplied.
- **KV-cache streaming** between pipeline-parallel stages.
- Any GPU-to-GPU **work queue** where the producer shouldn't pay host-launch
  latency per item.

---

# Goal

- Put lesson 9's ring buffer into NVSHMEM symmetric memory.
- See that the producer's `nvshmem_int_put` of the head counter is what
  "notifies" the consumer — no separate signal needed.
- Measure cross-GPU push/pop latency and compare to lesson 9's single-GPU
  version.

---

# Background

## What's symmetric, what's not

- `slots[cap]` — **symmetric**. Producer writes to `&slots[head%cap]` on the
  *consumer's* PE via `nvshmem_int_put`. (Equivalently, put to the consumer's
  copy of `slots`, same VA.)
- `head` — **symmetric**. Producer owns it; consumer reads it. Producer
  `nvshmem_int_put`s the new head to the consumer after a `quiet`.
- `tail` — **symmetric**. Consumer owns it; producer reads it (via
  `nvshmem_int_g`) to check for full.

## The producer push (device-side)

```c
__device__ void push(SymmRing r, int val, int cons_pe) {
    for (;;) {
        int h = r.head_local;                 // my own head
        int t = nvshmem_int_g(r.tail_ptr, cons_pe);  // remote tail
        if (h - t < r.cap) break;             // not full
    }
    nvshmem_int_put(&r.slots[h % r.cap], &val, 1, cons_pe);  // data -> consumer
    nvshmem_quiet();                          // data landed BEFORE head
    int newh = h + 1;
    nvshmem_int_put(r.head_ptr, &newh, 1, cons_pe);          // head -> consumer
    r.head_local = newh;
}
```

The `quiet` between the data put and the head put is the cross-GPU analog of
lesson 9's `__threadfence`. Without it, the consumer might see the new head
before the data — and read garbage.

## The consumer pop (device-side, on the consumer PE)

```c
__device__ int pop(SymmRing r) {
    for (;;) {
        int h = *r.head_ptr;        // head was put here by producer; local read
        int t = r.tail_local;
        if (h != t) break;          // not empty
    }
    int v = r.slots[t % r.cap];     // local read of symmetric slot
    r.tail_local = t + 1;
    // tail is informational for the producer; put it back so producer sees full
    int newt = t + 1;
    nvshmem_int_put(r.tail_ptr, &newt, 1, /*producer_pe=*/...);  // or use a shared atomic
    return v;
}
```

(We simplify the tail echo in the actual code; the key idea is that the
consumer reads `head` locally because the producer put it there.)

---

# Architecture Diagram

```
   Symmetric ring across PE0 (producer) and PE1 (consumer):

        PE 0                                  PE 1
   ┌──────────────┐                    ┌──────────────┐
   │ head_local   │  put head ───────▶ │ head (symm)  │ ◀ read by consumer
   │              │  put data ───────▶ │ slots[]      │ ◀ read by consumer
   │              │  get tail ◀─────── │ tail (symm)  │ ◀ written by consumer
   └──────────────┘                    └──────────────┘
        push loop                            pop loop
```

---

# Source Code Walkthrough

`nvshmem_ring_buffer.cu`:

- `struct SymmRing { int* slots; int* head; int* tail; int cap; }` — all three
  are symmetric pointers.
- `producer_kernel(ring, cons_pe, count)` — single-thread producer; pushes
  `count` values.
- `consumer_kernel(ring, prod_pe, out, count)` — single-thread consumer on the
  consumer PE; pops `count` values into `out`.
- `main()` — alloc symmetric, launch producer on PE 0 and consumer on PE 1
  (separate streams), sync, verify ordering on PE 1.

(See the file for the exact push/pop with `quiet` placement.)

---

# Build

```bash
cmake -S . -B build -DBUILD_NVSHMEM_LESSONS=ON -DNVSHMEM_DIR=/path/to/nvshmem
cmake --build build -j --target nvshmem_ring_buffer
```

---

# Run

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/lesson13-nvshmem-ring-buffer/nvshmem_ring_buffer
CUDA_VISIBLE_DEVICES=0,1 ./build/lesson13-nvshmem-ring-buffer/nvshmem_ring_buffer 8 10000
```

---

# Expected Output

```
==== lesson 13: NVSHMEM ring buffer ====
PE0=producer, PE1=consumer, capacity=8, count=10000

PE1 popped 10000 values
first 8: [0, 1, 2, 3, 4, 5, 6, 7]
in order: YES
elapsed 2.8 ms   (cross-GPU push/pop)
```

---

# Experiment

1. **Drop the quiet.** Remove the `nvshmem_quiet()` between data-put and
   head-put. The consumer will occasionally read stale slots. Reproduce.
2. **Capacity sweep.** At cap=1 you get strict ping-pong (every push waits for
   a pop). At cap=count, no backpressure. Find the throughput sweet spot.
3. **Batch the head advance.** Push B values per head increment (head counts
   batches). One `quiet` per B values. Measure the throughput jump — this is
   the single most important DeepEP optimization.
4. **Two channels.** Run two independent rings (PE0→PE1 and PE0→PE2) from the
   same producer kernel. They share the producer's bandwidth but double the
   consumer parallelism. DeepEP runs many.

---

# Performance Analysis

- **Per-item latency** is now bounded by `quiet` + two puts (data + head) ≈
  a few hundred ns on NVLink. Compare to lesson 9's single-GPU ~tens of ns —
  the cross-GPU tax is real but still ~10× below host-launch latency.
- **Batching kills the tax.** Pushing B items per `quiet` amortizes the
  ordering cost over B items; per-item cost approaches the raw NVLink
  per-byte cost. DeepEP's batches are hundreds to thousands of tokens.
- **Capacity hides producer/consumer speed mismatch.** If the consumer is
  briefly slow, the producer fills the buffer instead of stalling. This is
  why DeepEP sizes its channel buffers to absorb a few batches of slack.

---

# Exercises

1. **Variable-size items.** Push a `(size, payload)` pair so items can be
   different lengths. The consumer must read `size` first. (MoE dispatch
   sends variable token counts per expert — this is the shape.)
2. **MPSC.** Two producer PEs push to one consumer. The consumer needs an
   atomic head merge — or, simpler, two separate rings and the consumer
   polls both. DeepEP uses one consumer per channel to stay SPSC.
3. **Backpressure measurement.** Make the consumer deliberately slow and
   observe the producer's stall rate vs capacity. Plot throughput vs capacity.

---

# DeepEP Connection

```
Lesson 09  SPSC ring (single GPU, __threadfence)
Lesson 13  SPSC ring (NVSHMEM, nvshmem_quiet)        <- you are here
Lesson 14  producer/consumer + flag signaling
   ↓
DeepEP     ChannelBuffer:
             - slots in symmetric memory (today)
             - head/tail in symmetric memory, batched advances (today + ex.3)
             - N channels in parallel (lesson 17)
             - per-batch "tokens ready" flag (lesson 14)
```

When you open DeepEP's source in lesson 18, search for `nvshmem_quiet` inside
the dispatch kernel — every occurrence is the cross-GPU analog of today's
fence, guarding a head/batch advance. The structure is identical.
