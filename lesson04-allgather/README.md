# Lesson 04 — AllGather

> Every rank contributes a slice. Every rank ends up with the full
> concatenation. The reverse of broadcast — and the first collective where
> *every* rank is both a sender and a receiver.

---

# Overview

## What are we building?

An **AllGather** across `n` GPUs. Each rank `r` owns slice `r` of a logical
array. After AllGather, every rank owns the whole array: `[slice0, slice1, …,
slice_{n-1}]`.

We implement two schedules:

1. **Naive:** every rank sends its slice to every other rank directly
   (`n·(n-1)` copies, but `n-1` per rank).
2. **Ring:** rank `r` sends slice `(r-1) mod n` to rank `(r+1) mod n`, then
   forwards; after `n-1` steps every rank has all slices.

```
Before:   r0:[A]  r1:[B]  r2:[C]  r3:[D]
After:    r0:[ABCD] r1:[ABCD] r2:[ABCD] r3:[ABCD]
```

## Why does it matter?

AllGather is the **second half of AllReduce** (lesson 7): AllReduce =
ReduceScatter + AllGather. It is also the **collective TP uses for the
forward pass of a tensor-parallel linear layer** — each rank computes a shard
of the output, then AllGather concatenates them.

## Where is it used in LLM inference?

- **Tensor-Parallel forward:** after `Y = X @ W` where `W` is sharded across
  ranks, AllGather assembles the full `Y` so the next layer can use it.
- **ZeRO-3 / FSDP:** before a forward pass, AllGather reconstructs the full
  parameter shard.
- **DeepEP combine** (the reverse of dispatch): expert outputs are gathered
  back to the originating token positions.

---

# Goal

- Implement AllGather naive and ring with only `cudaMemcpyPeerAsync`.
- See the ring win for large payloads by pipelining.
- Understand the `n-1` step count and why each step moves `S/n` bytes per rank.

---

# Background

## The ring AllGather, step by step (n=4)

Initial: `r0:[A] r1:[B] r2:[C] r3:[D]`. Each rank has one slice; wants four.

```
step 0:  r0 sends A to r3 (wraps); r1 sends B to r0; r2 sends C to r1; r3 sends D to r2
         now: r0:[A,B] r1:[B,C] r2:[C,D] r3:[D,A]
step 1:  forward the new slice along the ring
         now: r0:[A,B,C] r1:[B,C,D] r2:[C,D,A] r3:[D,A,B]
step 2:  forward once more
         now: r0:[A,B,C,D] ... every rank complete
```

n-1 = 3 steps, each rank sends S/n bytes per step, total per-rank send =
`(n-1)·S/n ≈ S` bytes. That's bandwidth-optimal.

## Volume

| Algorithm | Per-rank sends | Per-rank receives |
|-----------|----------------|-------------------|
| Naive     | (n-1)·S/n      | (n-1)·S/n         |
| Ring      | (n-1)·S/n      | (n-1)·S/n         |

Same volume! The ring wins not by moving less data, but by **using every link
simultaneously** — naive's "every rank to every other rank" can't be issued all
at once without oversubscribing the links, so it serializes.

---

# Architecture Diagram

```
   Ring AllGather (n=4), arrows = "send my newest slice to the next rank"

      ┌───────┐  step0   ┌───────┐  step1   ┌───────┐  step2   ┌───────┐
      │  r0   │ ───────▶ │  r1   │ ───────▶ │  r2   │ ───────▶ │  r3   │
      │[A]    │          │[B]    │          │[C]    │          │[D]    │
      └───────┘          └───────┘          └───────┘          └───────┘
          ▲                                                       │
          └────────────────────── wraps ─────────────────────────┘
```

---

# Source Code Walkthrough

`allgather.cu`:

- Each rank `r` has a buffer of `n` slots; slot `r` is pre-filled with its
  slice. The other slots start zeroed.
- `allgather_naive` — for each `dst != src`, copy `slice[src]` from `d[src]`
  into `d[dst][src]`.
- `allgather_ring` — for `step = 0..n-2`: each rank `r` sends its current
  "head" slice to rank `(r+1) % n`, and receives into the next slot. We use a
  per-rank stream and a barrier between steps so a rank doesn't forward a
  slice it hasn't received yet.

Key shape (ring):

```c
for (int step = 0; step < n - 1; ++step) {
    int send_slice = (r - step + n) % n;   // the slice I'm sending this step
    int recv_slice = (r - step - 1 + n) % n;
    cudaMemcpyPeerAsync(d[r+1][send_slice], r+1,
                        d[r][send_slice],   r,   slice_bytes, streams[r]);
    // barrier so the forward next step sees the data
}
```

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target allgather
```

---

# Run

```bash
./build/lesson04-allgather/allgather
./build/lesson04-allgather/allgather 262144   # 1 MiB total
```

---

# Expected Output

```
==== lesson 04: allgather ====
n_gpus = 4
slice = 256 KiB ints, total = 1.00 MiB

before: r0[ 1..] r1[.. ] r2[.. ] r3[.. ]   (only own slice filled)

==== naive allgather ====
after:  r0[A B C D] r1[A B C D] r2[A B C D] r3[A B C D]  OK
0.42 ms

==== ring allgather ====
after:  r0[A B C D] r1[A B C D] r2[A B C D] r3[A B C D]  OK
0.18 ms   (3 steps, pipelined)
```

---

# Experiment

1. **Scale n.** At n=8, naive issues 56 copies; ring issues 7 steps. The gap
   widens.
2. **Scale S.** For very small S, naive may actually beat ring (less barrier
   overhead). Find the crossover.
3. **Break the barrier.** In the ring, remove the inter-step sync. Rank 1 will
   try to forward slice A before rank 0 has delivered it. Confirm corruption.
4. **Pipeline chunks.** Split each slice into chunks and forward chunk `c` as
   soon as it arrives while receiving chunk `c+1`. This turns the ring's
   per-step cost from `β·S/n` into `β·S/n / parallelism` — the real trick NCCL
   uses.

---

# Performance Analysis

- **Naive** saturates at the slowest single link because many ranks target the
  same destination simultaneously, forcing serialization.
- **Ring** keeps every link busy with exactly one transfer per step. Total time
  ≈ `(n-1)·(α + β·S/n)`. For large S the `β·S/n` term dominates per step, so
  ring ≈ `β·S·(n-1)/n` ≈ `β·S` — bandwidth-optimal.
- The **barrier between steps** adds `α` per step. Chunk pipelining (exercise 4)
  removes most of it by overlapping step `k`'s tail with step `k+1`'s head.

---

# Exercises

1. **Reverse it into ReduceScatter.** AllGather is "every rank gets all
   slices." ReduceScatter (lesson 6) is "every rank gets the *sum* of one
   slice." Reuse the ring, but instead of copying, accumulate at each hop.
2. **Make it async + overlapped.** Issue all `n-1` ring steps' copies without
   a full barrier, using events to express only the *necessary* dependencies
   (lesson 11). Watch the barrier overhead vanish.
3. **Compare to NCCL.** If you have NCCL available, time `ncclAllGather` on the
   same size/n and divide. NCCL should be 2–5× faster thanks to chunk
   pipelining and multiple channels.

---

# DeepEP Connection

```
Lesson 04  AllGather (ring)
   ↓
NCCL       all_gather() — ring + chunked + multi-channel
   ↓
DeepEP     combine phase — conceptually an AllGather-with-routing:
           each expert GPU sends its processed tokens back to the token's home
           GPU. Not a uniform AllGather (the destinations are token-dependent),
           but the *mechanics* (ring-buffered peer sends + completion signals)
           are identical to today's ring AllGather.
```

The DeepEP **combine** kernel is, mechanically, an AllGather where the slice
boundaries are decided per-token by the routing (lesson 15) rather than by
rank index. Today's ring is the uniform-routing special case.
