# Lesson 06 — ReduceScatter

> Reduce *and* scatter: each rank ends up owning one slice of the global sum.
> This is the first half of AllReduce — and the moment the ring algorithm
> truly starts to shine.

---

# Overview

## What are we building?

A **ReduceScatter** across `n` GPUs. Each rank `r` has a full buffer of length
`L = n · chunk`. After ReduceScatter, rank `r` owns chunk `r` of the
element-wise sum:

```
Before:  r0:[a0 a1 a2 a3]  r1:[b0 b1 b2 b3]  r2:[c0 c1 c2 c3]  r3:[d0 d1 d2 d3]
After:   r0:[a0+b0+c0+d0]  r1:[a1+b1+c1+d1]  r2:[a2+b2+c2+d2]  r3:[a3+b3+c3+d3]
                  chunk0            chunk1            chunk2            chunk3
```

We implement:

1. **Naive:** every rank sends every chunk to the rank that owns that chunk;
   owner sums. `n-1` receives per rank, each `S/n`.
2. **Ring ReduceScatter:** the canonical ring — `n-1` steps, each rank sends
   one chunk and receives+accumulates a different chunk. After `n-1` steps,
   each rank holds the full sum of its owned chunk.

## Why does it matter?

ReduceScatter is **half of AllReduce** (AllReduce = ReduceScatter +
AllGather). The ring ReduceScatter is the algorithm that makes AllReduce
bandwidth-optimal: every rank sends exactly `(n-1)·S/n ≈ S` bytes total, and
every link is busy every step. This is the algorithm DeepSeek (and everyone
else) uses for data-parallel gradient sync.

## Where is it used in LLM inference?

- **DP gradient sync** (training): the canonical ring ReduceScatter.
- **Sequence-parallel** reductions of attention norms.
- The **scatter half** of any fused AllReduce in a TP forward.

---

# Goal

- Implement ring ReduceScatter and *feel* the `n-1` step / `S/n` per step
  rhythm.
- See why the ring is bandwidth-optimal: per-rank send = `S·(n-1)/n`.
- Convince yourself AllReduce = ReduceScatter ⊕ AllGather (lesson 7 proves it).

---

# Background

## The ring ReduceScatter, n=4, chunks A,B,C,D per rank

Each rank initially holds `[A_r, B_r, C_r, D_r]` (its own version of every
chunk). Goal: rank 0 ends with `A_0+A_1+A_2+A_3`, rank 1 with `B_0+...`, etc.

```
step 0:  r sends chunk (r-1) mod n to r+1, receives chunk (r-2) mod n from r-1, adds
step 1:  forward the accumulated chunk
step 2:  forward once more  -> each rank holds the full sum of its owned chunk
```

After n-1 = 3 steps, each rank has accumulated `n` contributions into one
chunk. The *mechanics* are identical to ring AllGather (lesson 4) — the only
difference is the receiver **adds** instead of overwrites.

## Volume (the key table)

| Algorithm | Per-rank send | Per-rank receive |
|-----------|---------------|------------------|
| Naive     | (n-1)·S/n     | (n-1)·S/n        |
| Ring      | (n-1)·S/n     | (n-1)·S/n        |

Same volume as naive — but the ring keeps every link busy every step, so
wall-clock is `(n-1)·(α + β·S/n)`, vs naive's serialization on hot links.

---

# Architecture Diagram

```
   Ring ReduceScatter (n=4). Each arrow = "send one chunk; receiver adds it."

        ┌─ r0 ─┐ send    ┌─ r1 ─┐ send    ┌─ r2 ─┐ send    ┌─ r3 ─┐
        │chunks│ ──────▶ │chunks│ ──────▶ │chunks│ ──────▶ │chunks│
        └──────┘         └──────┘         └──────┘         └──────┘
            ▲                                                │
            └──────────── wraps (r3 -> r0) ──────────────────┘

   After n-1 steps, rank r holds the full sum of chunk r.
```

---

# Source Code Walkthrough

`reduce_scatter.cu`:

- `add_into` kernel from lesson 5, reused.
- `rs_naive` — for each `dst`, every `src != dst` copies its `dst`-chunk into
  `dst`'s scratch; `dst` runs `add_into`. `n-1` receives per rank.
- `rs_ring` — `n-1` steps. At step `k`, rank `r` sends the chunk it most
  recently accumulated to rank `(r+1) % n`, and receives a chunk from
  `(r-1) % n` into scratch, then adds. After the loop, rank `r`'s owned chunk
  holds the full sum.

Key shape (ring):

```c
int owned = r;  // chunk index this rank will end up owning
for (int step = 0; step < n - 1; ++step) {
    int send_chunk = (owned - step + n) % n;     // chunk I forward this step
    int recv_chunk = (owned - step - 1 + n) % n; // chunk I receive & add
    cudaMemcpyPeerAsync(d[next] + recv_chunk, next,
                        d[r]     + send_chunk, r, chunk_bytes, streams[r]);
    // barrier, then receiver adds scratch into its recv_chunk slot
}
```

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target reduce_scatter
```

---

# Run

```bash
./build/lesson06-reduce-scatter/reduce_scatter
./build/lesson06-reduce-scatter/reduce_scatter 262144
```

---

# Expected Output

```
==== lesson 06: reduce-scatter ====
n_gpus = 4
L = 1048576 ints (4.00 MiB per rank), chunk = 1.00 MiB
rank r holds value r at every index -> chunk sum = 0+1+2+3 = 6

==== naive reduce-scatter ====
rank0 chunk0[0..3] = [6,6,6,6] ... OK
0.62 ms

==== ring reduce-scatter ====
rank0 chunk0[0..3] = [6,6,6,6] ... OK
0.24 ms   (3 steps, pipelined)
```

---

# Experiment

1. **Then AllGather it.** Take the ring ReduceScatter output and run lesson 4's
   ring AllGather on it. You've just built a ring AllReduce (lesson 7 does
   this explicitly).
2. **Vary chunk count.** With `n` chunks the ring does `n-1` steps. Try n=2,4,8
   and confirm step count.
3. **Pipeline chunks.** Split each chunk into sub-chunks and pipeline. The
   per-step cost drops from `β·S/n` toward `β·S/n / k` for `k` sub-chunks —
   this is what makes the ring actually beat the naive at scale.
4. **Wrong-op bug.** Swap `add_into` for a plain copy in the ring. Now each
   rank ends up with only the *last* contributor's chunk, not the sum.
   Confirm the bug, then restore.

---

# Performance Analysis

- **Naive** oversubscribes the destination ranks' ingress: all `n-1` senders
  target rank `r` for chunk `r`. On a mesh, that's `n-1` simultaneous receives
  into one GPU — serialized by the receiver's ingress bandwidth.
- **Ring** spreads the load: at every step, *every* rank is both sender and
  receiver of exactly one chunk, on a distinct link. Total time
  `≈ (n-1)·(α + β·S/n)`. As `n` grows, per-step work shrinks (`S/n`), so the
  ring *scales*: total time stays `≈ β·S` regardless of `n`.
- This `β·S`-independence of `n` is the **bandwidth optimality** everyone
  talks about. Naive's time grows with `n`; the ring's doesn't.

---

# Exercises

1. **Fuse the add into the copy.** Instead of copy-to-scratch-then-`add_into`,
   write a receiver kernel that reads the sender's chunk over UVA/P2P and adds
   directly into the destination chunk. One kernel per step, no scratch.
2. **Reverse the ring.** Run the ring backwards. You've just invented
   AllGather-from-Ring — confirming ReduceScatter and AllGather are duals.
3. **Async + events.** Replace the per-step `cudaDeviceSynchronize` barriers
   with per-pair events (lesson 11). The ring's steps can now overlap across
   ranks.

---

# DeepEP Connection

```
Lesson 06  ring ReduceScatter
   ↓
NCCL       reduce_scatter() — ring, chunked, fused add
   ↓
DeepEP     the dispatch routing math: DeepEP computes, up front, the
           per-(src,dst) token count — the equivalent of knowing "how big is
           each chunk this rank must send/receive" before issuing the copies.
           Today's chunk size is uniform; DeepEP's is a sparse, per-pair matrix
           (lesson 15). The *ring-buffered issue pattern* is the same.
```

DeepEP's dispatch is, in essence, a **non-uniform ReduceScatter/AllToAll**: the
amount each rank sends to each other rank is decided by the router, not fixed
at `S/n`. The hardware primitive (ring-buffered peer writes with add-at-
destination) is identical to today's ring.
