# Lesson 03 — Broadcast

> Rank 0 has a buffer. Every rank needs the same buffer. We build a Broadcast
> three ways — naive, ring, tree — and see why naive is `O(n)` latency while
> ring and tree are `O(log n)`.

---

# Overview

## What are we building?

A hand-rolled **Broadcast** across `n` GPUs, where rank 0 sends its entire
buffer to every other rank. We implement and compare:

1. **Naive:** rank 0 sends a separate peer copy to each of ranks 1..n-1,
   **serialized** on one stream.
2. **Ring:** data flows 0→1→2→…→n-1; each rank forwards what it just received.
3. **Tree (binary):** rank 0 sends to rank 1; then 0→2 and 1→3; then 0→4, …
   doubling the set of "have-it" ranks each step.

```
Naive (serialized):         Ring:                      Tree (binary):
  0 -> 1                     0 -> 1 -> 2 -> ... -> n-1   step0: 0 -> 1
  0 -> 2                                                 step1: 0 -> 2, 1 -> 3
  0 -> 3                                                 step2: 0 -> 4, 1->5,2->6,3->7
  ...
  0 -> n-1                  n-1 hops, but pipelined      log2(n) steps
  n-1 serialized hops
```

## Why does it matter?

Broadcast is the simplest collective that has a **real algorithmic choice**.
The naive version is what you'd write if you didn't think; the ring and tree
are what NCCL actually does. The lesson is that **the same physical primitive
(lesson 2's peer copy) gives wildly different latency depending on the
schedule**.

## Where is it used in LLM inference?

- Broadcasting model **weights** from the data-loading rank to all TP ranks at
  startup.
- Broadcasting the **same input embeddings** to the first TP rank of every
  pipeline stage in a batched decode.
- Broadcasting **expert bias / scale** tensors in MoE.

---

# Goal

- Implement broadcast three ways using only `cudaMemcpyPeerAsync`.
- Explain why naive is `O(n)` latency but ring/tree are `O(log n)`.
- Predict which one wins for *small* messages (tree) vs *large* (ring).

---

# Background

## Latency vs bandwidth, the broadcast edition

For a message of size `S` and per-hop latency `α` plus per-byte cost `β`:

| Algorithm | Steps | Time (model)        | Best when           |
|-----------|-------|---------------------|---------------------|
| Naive     | n-1   | (n-1)·(α + β·S)     | never               |
| Ring      | n-1   | (n-1)·α + β·S       | large S (pipeline)  |
| Tree      | log₂n | log₂n·(α + β·S)     | small S (few hops)  |

The ring's magic: although it has n-1 *hops*, only **one copy** is in flight
per hop at a time *per chunk*, so for large S you can pipeline chunks and the
β·S term doesn't multiply by n-1. The tree's magic: only `log₂n` hops, so for
small S where `α` dominates, tree wins.

## NCCL's actual choice

NCCL uses a **tree for small messages** (latency-dominated) and **rings for
large messages** (bandwidth-dominated), and switches at a size threshold it
auto-tunes. It also runs **multiple trees/rings in parallel** (channels) to
saturate links. We'll meet channels in lesson 19.

---

# Architecture Diagram

```
Rank:    0    1    2    3        (n=4)

Naive:   0 ──> 1
         0 ──> 2          (3 serial hops)
         0 ──> 3

Ring:    0 ──> 1 ──> 2 ──> 3     (3 hops, but each hop is full-size)

Tree:    0 ──> 1                 step 0
         0 ──> 2                 step 1   (1 also has it; could send to 3)
         1 ──> 3                 step 2 (parallel with above)
         ⇒ 2 steps for n=4
```

---

# Source Code Walkthrough

`broadcast.cu`:

- `broadcast_naive(streams, d_buf, bytes, n)` — loop `dst=1..n-1`, issue
  `cudaMemcpyPeerAsync(d[dst], dst, d[0], 0, bytes, streams[0])` on a single
  stream. Serialized.
- `broadcast_ring(...)` — same loop, but *each* forward is on the *next* rank's
  stream with an event dependency so they can overlap on NVSwitch.
- `broadcast_tree(...)` — `for step in 0..log2(n):` every rank `r` that has the
  data and whose partner `r | (1<<step)` exists sends to it. All sends in a
  step are concurrent.

Key shape (tree):

```c
for (int step = 0; (1 << step) < n; ++step) {
    for (int r = 0; r < n; ++r) {
        int partner = r | (1 << step);
        if (r < partner && partner < n) {
            // r has the data, partner doesn't yet -> r sends to partner
            cudaMemcpyPeerAsync(d[partner], partner, d[r], r, bytes, streams[r]);
        }
    }
    cudaDeviceSynchronize();  // barrier between steps
}
```

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target broadcast
```

---

# Run

```bash
./build/lesson03-broadcast/broadcast
./build/lesson03-broadcast/broadcast 1048576   # 1 MiB of ints
```

---

# Expected Output

```
==== lesson 03: broadcast ====
n_gpus = 4
bytes = 4.00 MiB
src[0..3] = [1, 2, 3, 4]

==== naive broadcast ====
all ranks match src: YES
time 0.85 ms  (~5 GB/s aggregate, 3 serial hops)

==== ring broadcast ====
all ranks match src: YES
time 0.31 ms  (~14 GB/s — pipelined)

==== tree broadcast ====
all ranks match src: YES
time 0.22 ms  (~19 GB/s — 2 hops for n=4)
```

For small messages the tree should win; for very large messages ring pulls
ahead because it pipelines the payload across hops.

---

# Experiment

1. **Vary n.** With n=2,4,8, plot naive vs ring vs tree time. Naive should
   scale ~linearly; tree ~logarithmically.
2. **Vary S.** Fix n=4. Sweep S from 1 KiB to 256 MiB. Find the crossover
   where ring overtakes tree. That crossover is exactly the threshold NCCL
   auto-tunes.
3. **Break the tree.** Remove the `cudaDeviceSynchronize()` between steps.
   Tree now races — some partners receive before their sender has the data.
   Confirm the corruption, then put it back. This is why collectives need
   barriers *between* dependency steps but not *within* a pipelined chunk.

---

# Performance Analysis

- **Naive** is bottlenecked by rank 0's single egress link: it must send
  `(n-1)·S` bytes, all serialized. Time ≈ `(n-1)·S / BW_one_link`.
- **Ring** uses every link simultaneously (rank 0 sends once, rank 1 forwards
  once, …), so the per-hop work is `S`, and with chunk pipelining the total is
  ≈ `(n-1)·α + S·(n-1)/n / BW`. For large S this approaches `S/BW` —
  bandwidth-optimal.
- **Tree** uses `log₂n` steps; each step every active link fires. For small S
  where `α` dominates, total ≈ `log₂n · (α + β·S)` — the fewest hops, hence
  the lowest latency.

---

# Exercises

1. **Make it a Scatter.** Instead of every rank receiving the *full* buffer,
   rank 0 splits its buffer into n equal slices and rank k receives slice k.
   (Hint: only the offset and size change.)
2. **Pipeline the ring.** Split S into chunks and have each rank forward chunk
   c as soon as it arrives, overlapping with receiving chunk c+1. This is the
   real ring algorithm — and the direct precursor to lesson 7's ring
   AllReduce.
3. **Measure the host-vs-device-launch gap.** For very small S, the
  `cudaMemcpyPeerAsync` API call itself (~5 µs) dominates. Use a kernel that
   issues the copy from the device side (a hint of NVSHMEM, lesson 12) and
   watch the latency drop.

---

# DeepEP Connection

```
Lesson 03  broadcast (tree for small, ring for large)
   ↓
NCCL       bcast() — same tree/ring choice, auto-tuned
   ↓
DeepEP     broadcasts expert weight shards at layer init;
           the dispatch itself is closer to AllToAll (lesson 8), but the
           *signaling* of "this token is ready" uses tree-like fan-out.
```

The tree-vs-ring decision NCCL makes here is the same decision DeepEP makes
when choosing its **normal-latency** (tree-like, few hops) vs **low-latency**
(ring-buffered, many small parallel transfers) dispatch kernels — see lessons
17 and 19.


你的理解**方向是对的**，但有两个地方需要纠正一下：

1. **不是开了多个 stream 进程**（CUDA Stream 不是进程，也不是线程）
2. **也不是 CUDA 自己随意调度**，而是**通过 Event 建立了明确的执行依赖关系**，CUDA Runtime 根据这些依赖来安排执行。

---

## 可以理解成下面这个模型

每个 GPU 都有一个 Stream：

```text
GPU0 : stream0
GPU1 : stream1
GPU2 : stream2
GPU3 : stream3
```

整个循环实际上是在**提前把所有工作都提交（enqueue）到各个 Stream 中**。

例如对于 4 张 GPU、2 个 chunk：

GPU0 的 stream：

```text
Wait ready[0][0]
Memcpy chunk0 -> GPU1
Record ready[1][0]

Wait ready[0][1]
Memcpy chunk1 -> GPU1
Record ready[1][1]
```

GPU1 的 stream：

```text
Wait ready[1][0]
Memcpy chunk0 -> GPU2
Record ready[2][0]

Wait ready[1][1]
Memcpy chunk1 -> GPU2
Record ready[2][1]
```

GPU2 的 stream：

```text
Wait ready[2][0]
Memcpy chunk0 -> GPU3
Record ready[3][0]

Wait ready[2][1]
Memcpy chunk1 -> GPU3
Record ready[3][1]
```

注意：

CPU 很快就把这些命令全部提交完了，并不会等待。

---

## Event 就像"红绿灯"

例如：

```cpp
cudaStreamWaitEvent(streams[1], ready[1][0], 0);
```

表示：

> GPU1 的 stream 执行到这里时，如果 `ready[1][0]` 没完成，就停在这里。

什么时候继续？

就是：

```cpp
cudaEventRecord(ready[1][0], streams[0]);
```

真正执行完成之后。

于是：

```text
GPU0 stream

Memcpy chunk0
↓

Record ready[1][0]
          │
          ▼
GPU1 stream 被唤醒
↓

Memcpy chunk0
↓

Record ready[2][0]
          │
          ▼
GPU2 stream 被唤醒
```

整个过程都是 GPU 自己完成同步，不需要 CPU 再参与。

---

## 为什么效率高？

如果没有 Event：

```text
CPU：

GPU0->GPU1

等待

GPU1->GPU2

等待

GPU2->GPU3
```

CPU 必须不断同步。

而现在：

CPU 一次性提交：

```text
GPU0 stream
GPU1 stream
GPU2 stream
```

全部命令。

然后：

```text
GPU 自己等 Event

GPU 自己开始下一步

GPU 自己通知下一张卡
```

CPU 基本可以退出了，只在最后：

```cpp
cudaStreamSynchronize(...)
```

等待全部结束。

---

## 更准确地说，这段代码实现的是一个**数据流图（DAG）**

可以画成这样：

```text
GPU0
Chunk0 Copy
     │
     ▼
ready[1][0]
     │
     ▼
GPU1
Chunk0 Copy
     │
     ▼
ready[2][0]
     │
     ▼
GPU2
Chunk0 Copy
```

Chunk1 又是另一条链：

```text
GPU0
Chunk1 Copy
     │
     ▼
ready[1][1]
     │
     ▼
GPU1
Chunk1 Copy
```

所以实际上存在很多条这样的依赖链：

```text
Chunk0：GPU0 → GPU1 → GPU2 → GPU3
Chunk1：GPU0 → GPU1 → GPU2 → GPU3
Chunk2：GPU0 → GPU1 → GPU2 → GPU3
...
```

CUDA Runtime 根据这些 Event 自动调度，只要某个 Event 满足，它后面的操作立即开始执行。

---

### 一句话总结

你的理解可以修正为：

> **这段代码不是开启多个“进程”，而是为每个 GPU 准备了一个 Stream，并一次性把所有 chunk 的复制任务都提交到这些 Stream 中。`cudaEventRecord` 和 `cudaStreamWaitEvent` 为不同 GPU、不同 chunk 建立了依赖关系。之后 CUDA Runtime 根据这些 Event 自动唤醒后续操作，实现了跨 GPU 的流水线（pipeline）执行，而 CPU 无需逐步介入调度。**
