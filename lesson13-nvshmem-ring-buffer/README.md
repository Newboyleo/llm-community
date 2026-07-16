# Lesson 13 — NVSHMEM Ring Buffer

> Lesson 9's SPSC ring buffer, reborn across GPUs. The producer echoes head
> to the consumer PE, the consumer echoes tail to the producer PE, and the
> slots live in symmetric memory.
> A kernel on the producer streams tokens straight into the consumer's HBM.

---

# Overview

## What are we building?

An SPSC ring buffer where **producer and consumer are on different GPUs**.
Producer PE 0 pushes values that land in PE 1's symmetric buffer; consumer PE
1 pops them. Coordination uses put-based cursor mirrors: producer puts `head`
to PE 1, consumer puts `tail` back to PE 0, and each side waits on its local
symmetric copy. The data put is followed by an `nvshmem_quiet` before `head`
advances.

```
   PE 0 (producer)                          PE 1 (consumer)
   ┌──────────────┐  nvshmem_int_put        ┌──────────────┐
   │ push loop:   │ ──────────────────────▶ │ slots[]      │
   │  write slot  │   (data, into PE1 HBM)  │ (symmetric)  │
   │  quiet       │                         │              │
   │  head++      │ ──────────────────────▶ │ head (symm)  │
   │              │   (head, into PE1)      │              │
   │ tail mirror  │ ◀────────────────────── │  tail++      │
   └──────────────┘   (tail, into PE0)      │              │
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
- `head` — **symmetric**. Producer owns the logical counter; consumer reads its
  local PE 1 mirror after the producer puts updates there. Producer
  `nvshmem_int_put`s the new head to the consumer after a `quiet`.
- `tail` — **symmetric**. Consumer owns the logical counter; producer reads its
  local PE 0 mirror after the consumer puts updates back to the producer PE.
  This lesson does not remote-get `tail` from PE 1.

## The producer push (device-side)

```c
__device__ void push(SymmRing r, int val, int cons_pe) {
    for (;;) {
        int h = r.head_local;                 // my own head
        int t = *r.tail_ptr;                  // local PE0 tail mirror
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
    // tail is informational for the producer; put it to PE0's local mirror
    int newt = t + 1;
    nvshmem_int_put(r.tail_ptr, &newt, 1, /*producer_pe=*/...);  // or use a shared atomic
    return v;
}
```

(The key idea is symmetry of direction: consumer reads `head` locally because
the producer put it to PE 1; producer reads `tail` locally because the consumer
put it to PE 0.)

---

# Architecture Diagram

```
   Symmetric ring across PE0 (producer) and PE1 (consumer):

        PE 0                                  PE 1
   ┌──────────────┐                    ┌──────────────┐
   │ head_local   │  put head ───────▶ │ head (symm)  │ ◀ read by consumer
   │              │  put data ───────▶ │ slots[]      │ ◀ read by consumer
   │ tail (symm)  │ ◀ put tail ─────── │              │ ◀ written by consumer
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
CUDA_VISIBLE_DEVICES=0,1 NVSHMEM_REMOTE_TRANSPORT=none \
    /usr/bin/nvshmem_12/nvshmrun -np 2 \
    ./build/lesson13-nvshmem-ring-buffer/nvshmem_ring_buffer
CUDA_VISIBLE_DEVICES=0,1 NVSHMEM_REMOTE_TRANSPORT=none \
    /usr/bin/nvshmem_12/nvshmrun -np 2 \
    ./build/lesson13-nvshmem-ring-buffer/nvshmem_ring_buffer 8 10000
```

`NVSHMEM_REMOTE_TRANSPORT=none` keeps this single-node lesson on the local GPU
P2P path. Without it, NVSHMEM may try the IB/RDMA transport first; on machines
without `nvidia_peermem`/`nv_peer_mem`, that path can report a DMA-BUF failure
and stall during startup.

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


为什么 NVSHMEM 愿意接受这种浪费？
这是一个典型的空间换时间（space-for-time）的设计。
它换来了几个非常重要的优势：
统一地址模型：所有 PE 的对称对象具有一致的布局，不需要交换地址。
零握手通信：nvshmem_put(ptr, ..., pe) 不需要事先查询远端地址。
GPU 内直接通信：Kernel 可以直接发起 put/get，不需要 CPU 参与。
实现简单且高性能：运行时只需根据 pe 和对象偏移计算远端地址。
因此，NVSHMEM 更关注通信效率和编程模型，而不是把对称堆的内存利用率做到极致。
总结
所以你的理解基本是正确的，但可以更准确地表述为：
是的，nvshmem_malloc 分配的是对称内存，所有参与的 GPU 都必须分配相同大小、相同布局的空间。
是的，如果某些 GPU 永远不会使用某块对称内存，那么这部分空间确实会被浪费。
但在多数 HPC 和分布式 AI 场景中，每个 GPU 本来就需要自己的数据区（例如本地 KV Cache、消息队列、工作缓冲区等），因此这种额外浪费通常并不大，而换来的统一地址空间和极低通信开销是非常值得的。