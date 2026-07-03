# Lesson 07 — Ring AllReduce

> The capstone of the collectives arc. AllReduce = ReduceScatter + AllGather,
> and the ring makes it **bandwidth-optimal**: each rank sends only
> `S·(n-1)/n` bytes, independent of `n`.

---

# Overview

## What are we building?

A **Ring AllReduce** across `n` GPUs. Every rank ends up with the element-wise
sum of all ranks' buffers. We compose it from the two algorithms you already
built:

```
   AllReduce = ring ReduceScatter (lesson 6)   // each rank owns 1/n of the sum
            + ring AllGather    (lesson 4)     // every rank gets the full sum
```

We also compare against a **naive** AllReduce (reduce-to-zero + broadcast) to
make the bandwidth argument concrete.

## Why does it matter?

Ring AllReduce is *the* algorithm behind data-parallel training (the
"allreduce" in every DDP/FSDP step). Its claim to fame — that the per-rank
communication volume is `S·(n-1)/n ≈ S`, *not* `S·(n-1)` — is what lets you
train on 64 GPUs with barely more interconnect traffic than on 4. If you
understand one algorithm in this whole course, understand this one.

## Where is it used in LLM inference?

- **DP gradient sync** during training of the LLM you later inference.
- **ZeRO-1/2** parameter/gradient sharding reduce-scatters.
- The mental model underlies DeepEP's combine (which is an AllReduce-like
  gather-and-sum over experts).

---

# Goal

- Implement AllReduce as RS + AG and verify it equals the brute-force sum.
- Derive the per-rank volume `S·(n-1)/n` and confirm it empirically.
- Explain the latency term `(n-1)·α` and why NCCL adds a tree on top.

---

# Background

## The bandwidth argument (the whole point)

Naive AllReduce (reduce to rank 0, then broadcast) makes rank 0 ingest
`(n-1)·S` bytes and emit `(n-1)·S` bytes. Rank 0's links are the bottleneck;
adding more ranks doesn't help — it hurts.

Ring AllReduce splits the work into two phases:

| Phase          | Steps   | Per-rank send | Per-rank send total |
|----------------|---------|---------------|---------------------|
| ReduceScatter  | n-1     | S/n per step  | S·(n-1)/n           |
| AllGather      | n-1     | S/n per step  | S·(n-1)/n           |
| **Total**      | **2(n-1)** |            | **2·S·(n-1)/n ≈ 2S**|

Per-rank volume is `≈ 2S`, **independent of n**. Doubling the GPU count does
*not* double the traffic — it just splits the same `2S` across more links.
That's bandwidth optimality.

## The latency catch

Total time ≈ `2·(n-1)·α + 2·S·(n-1)/n / BW`. The first term grows with `n`. For
**small** `S`, this `(n-1)·α` latency dominates and the ring is *worse* than a
tree. NCCL therefore runs:

- a **ring** for the bandwidth-bound (large `S`) regime, and
- a **tree** (latency `O(log n)`) for the latency-bound (small `S`) regime,
  often fused so the ring handles bulk and the tree handles the residual.

---

# Architecture Diagram

```
   Ring AllReduce (n=4), buffer split into 4 chunks A,B,C,D per rank.

   Phase 1 — ReduceScatter (n-1 = 3 steps):
        r0 ─A─▶ r1 ─?─▶ r2 ─?─▶ r3 ─?─▶ (wrap)
        After 3 steps: r0 owns ΣA, r1 owns ΣB, r2 owns ΣC, r3 owns ΣD.

   Phase 2 — AllGather (n-1 = 3 steps):
        r0 ─ΣA─▶ r1 ─ΣB─▶ r2 ─ΣC─▶ r3 ─ΣD─▶ (wrap)
        After 3 steps: every rank owns [ΣA, ΣB, ΣC, ΣD].

   Total: 2(n-1) = 6 steps; per-rank send = 6 · S/4 = 1.5·S ≈ 2S·(n-1)/n.
```

---

# Source Code Walkthrough

`ring_allreduce.cu`:

- Reuses `add_into` (lesson 5).
- `ring_reduce_scatter(state)` — exactly lesson 6's ring, leaving each rank
  with the summed slice it owns.
- `ring_all_gather(state)` — exactly lesson 4's ring, but the "slice" each
  rank forwards is now its *summed* owned chunk. After this, every rank has
  all summed chunks.
- `naive_allreduce` — reduce-to-zero (lesson 5 naive) + broadcast (lesson 3
  naive). For comparison.

The whole AllReduce is literally:

```c
ring_reduce_scatter(s);   // phase 1
ring_all_gather(s);       // phase 2
```

— two functions you already wrote. That's the payoff of the incremental build.

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target ring_allreduce
```

---

# Run

```bash
./build/lesson07-ring-allreduce/ring_allreduce
./build/lesson07-ring-allreduce/ring_allreduce 262144
```

---

# Expected Output

```
==== lesson 07: ring allreduce ====
n_gpus = 4
L = 1048576 ints (4.00 MiB per rank)
expected (every index) = 6

==== naive allreduce (reduce-to-zero + broadcast) ====
rank0[0..3] = [6,6,6,6] ... OK
1.04 ms   (rank0 ingests + emits 3·S)

==== ring allreduce (RS + AG) ====
rank0[0..3] = [6,6,6,6] ... OK
0.46 ms   (6 steps, per-rank send ≈ 1.5·S)
```

---

# Experiment

1. **Vary n, hold S.** At n=2,4,8, ring time should stay ~flat (grows only in
   the `α·(n-1)` term), while naive grows ~linearly. Plot it.
2. **Vary S, hold n.** For tiny S the ring's `2(n-1)·α` latency may lose to
   naive's `2α + β·(n-1)·S`. Find the crossover — it's where NCCL switches
   algorithms.
3. **Chunk pipelining.** Split each `S/n` chunk into `k` sub-chunks and
   pipeline. The `(n-1)·α` term doesn't change, but the `β·S` term overlaps
   with compute. This is the single biggest ring optimization.
4. **Run it through NCCL.** Time `ncclAllReduce` on the same problem. Expect
   2–4× faster (fused kernels, channels, tuned chunk size). The gap *is* the
   content of lesson 19.

---

# Performance Analysis

- **Naive** time ≈ `2·(n-1)·S / BW_rank0`. Rank 0 is the bottleneck; the
  algorithm does *not* scale with `n`.
- **Ring** time ≈ `2·(n-1)·α + 2·S·(n-1)/n / BW`. The bandwidth term is
  `≈ 2S/BW` — independent of `n`. The latency term `2(n-1)·α` is the ring's
  weakness, fixed only by trees (lesson 3) or by hiding it under compute
  (overlap).
- **DeepSeek/NCCL improvement:** run **two rings in opposite directions**
  simultaneously (one for RS, one for AG, or both halves of the ring). This
  doubles link utilization and is standard. Our lesson uses one direction for
  clarity.

---

# Exercises

1. **Bidirectional ring.** Run RS clockwise and AG counter-clockwise
   simultaneously. Aggregate bandwidth should ~double.
2. **Add a tree tail.** After RS, instead of a ring AllGather, do a tree
   broadcast from each chunk-owner. For small `S` this is faster (fewer hops).
   This is the NCCL "ring + tree" hybrid.
3. **Overlap with compute.** Issue the RS, and while it runs do some dummy
   compute on a separate stream. Measure the hidden time. (This is
   "compute-communication overlap," the goal of every fused AllReduce in
   training frameworks.)

---

# DeepEP Connection

```
Lesson 07  Ring AllReduce (RS + AG, bandwidth-optimal)
   ↓
NCCL       all_reduce() — ring + tree hybrid, multi-channel, fused
   ↓
DeepEP     dispatch+combine is NOT an AllReduce, but inherits its two-phase
           structure:
             phase 1 (dispatch): a routed AllToAll  — like an "AllScatter"
             phase 2 (combine) : a routed AllToAll  — like an "AllGather"
           The ring-buffered, channel-parallel issue pattern is shared with
           the ring AllReduce. The difference is the *routing matrix* is
           per-token sparse, not uniform S/n.
```

The deep structural kinship: **AllReduce is the uniform-routing special case
of DeepEP's dispatch+combine.** Both are "every rank sends a piece to every
other rank, then everyone gets a reduced/combined result." Today the pieces
are equal-sized and the combine is sum; in DeepEP the pieces are
token-count-weighted and the combine is a weighted expert sum. Same skeleton.
