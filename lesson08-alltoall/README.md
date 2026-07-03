# Lesson 08 — AllToAll

> The collective at the heart of MoE. Rank `i` sends a *different* block to
> rank `j` for every pair `(i,j)`. Unlike AllGather (same data to all), every
> block is distinct.

---

# Overview

## What are we building?

An **AllToAll** across `n` GPUs. Each rank `i` has `n` blocks, one per
destination. After AllToAll, rank `j` has the `j`-th block from *every* rank
`i`. It is the **transpose** of the per-rank block layout.

```
Before (rank i has n blocks, block j is destined for rank j):
   r0:[ .0→0 .0→1 .0→2 .0→3 ]
   r1:[ .1→0 .1→1 .1→2 .1→3 ]
   r2:[ .2→0 .2→1 .2→2 .2→3 ]
   r3:[ .3→0 .3→1 .3→2 .3→3 ]

After (rank j has block j from every rank i):
   r0:[ .0→0 .1→0 .2→0 .3→0 ]
   r1:[ .0→1 .1→1 .2→1 .3→1 ]
   ...
```

We implement two forms:

1. **Naive (index transpose):** every rank sends its block `j` to rank `j`.
   This is `n·(n-1)` peer copies, but `n-1` sends and `n-1` receives per rank.
2. **In-place kernel transpose:** a single kernel that transposes the block
   matrix in HBM by reading `(i,j)` and writing `(j,i)` — but only works
   *within one GPU*. (Shown for contrast: AllToAll is fundamentally
   multi-GPU.)

## Why does it matter?

AllToAll is **the** communication primitive of Mixture-of-Experts. In MoE,
each token must reach its assigned expert(s), which live on specific GPUs. The
dispatch step is an AllToAll where block `(i,j)` contains "tokens on GPU i
whose chosen expert is on GPU j." The combine step is the reverse AllToAll.

If you understand AllToAll, you understand the skeleton of MoE dispatch.

## Where is it used in LLM inference?

- **MoE dispatch** (tokens → experts) and **combine** (expert outputs →
  tokens). Lessons 16–17 build this up.
- **Sequence parallelism** re-partitioning: reshuffling tokens across GPUs.
- **Pipeline parallelism** micro-batch handoff (a degenerate AllToAll along a
  chain).

---

# Goal

- Implement AllToAll with `cudaMemcpyPeerAsync`.
- See the `n²` block structure and that per-rank volume is `(n-1)·S/n` (same as
  AllGather!) — but the *routing* is per-pair, not per-ring.
- Understand why AllToAll is harder to make bandwidth-optimal than AllReduce:
  the per-pair blocks are independent, so there's no neat ring to pipeline.

---

# Background

## AllToAll vs AllGather: same volume, different shape

| Collective | Per-rank send | Pattern                        |
|------------|---------------|--------------------------------|
| AllGather  | (n-1)·S/n     | one slice to all (replicated)  |
| AllToAll   | (n-1)·S/n     | distinct block to each         |

Same *volume*, totally different *routing*. AllGather's ring works because
each slice is identical for every receiver — you forward the same bytes. In
AllToAll every block is unique to its `(src,dst)` pair, so the ring trick
doesn't directly apply. NCCL's AllToAll uses a different schedule (often a
"halving-doubling" or pipelined pairwise exchange).

## The MoE dispatch = AllToAll + routing

In uniform AllToAll, block `(i,j)` is `S/n` bytes for all `(i,j)`. In MoE
dispatch, block `(i,j)` has **variable size** = `count[i][j] · hidden_dim`,
where `count[i][j]` is the number of tokens on GPU i routed to an expert on
GPU j. That count matrix (lesson 15) is what makes MoE dispatch a
*non-uniform* AllToAll — and what DeepEP is optimized for.

---

# Architecture Diagram

```
   AllToAll (n=4). Each arrow = one distinct block, src->dst.

        ┌──── r0 ────┐  ┌──── r1 ────┐  ┌──── r2 ────┐  ┌──── r3 ────┐
   r0   │  block 0,0 │  │  block 0,1 │  │  block 0,2 │  │  block 0,3 │
        └────────────┘  └────────────┘  └────────────┘  └────────────┘
                          ▲               ▲               ▲
   r1 ...                 │ ...           │               │
        (every pair (i,j): rank i sends block (i,j) to rank j)
```

After: rank j's buffer = [ block(0,j), block(1,j), ..., block(n-1,j) ].

---

# Source Code Walkthrough

`alltoall.cu`:

- Layout: `d[r]` is `n·block` ints. Slot `j` on rank `r` is the block `r→j`
  (the block rank r sends to rank j). Pre-filled with a recognizable tag
  `10000*r + j` so we can verify.
- `a2a_naive` — for each `(src,dst)` with `src!=dst`: peer-copy `d[src][dst]`
  into `d[dst][src]`. (Slot `dst` of `d[src]` lands in slot `src` of `d[dst]`
  — that's the transpose.)
- After the call, rank `j`'s slot `i` should equal `10000*i + j`.

Key shape:

```c
for (int src = 0; src < n; ++src)
    for (int dst = 0; dst < n; ++dst)
        if (src != dst)
            cudaMemcpyPeerAsync(d[dst] + src*block, dst,
                                d[src] + dst*block, src, block_bytes, streams[src]);
```

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target alltoall
```

---

# Run

```bash
./build/lesson08-alltoall/alltoall
./build/lesson08-alltoall/alltoall 65536   # block size in ints
```

---

# Expected Output

```
==== lesson 08: alltoall ====
n_gpus = 4
block = 65536 ints (256 KiB), total per rank = 1.00 MiB

before: r0 slot0=0 slot1=1 slot2=2 slot3=3   (tag = 10000*r + slot)

==== naive alltoall ====
after:  r0 slot0=0 slot1=10000 slot2=20000 slot3=30000   (now tag = 10000*src + 0)
OK
0.39 ms
```

(Each rank's slot `i` now holds rank `i`'s contribution — the transpose.)

---

# Experiment

1. **Variable block sizes (mini MoE).** Make `block_size[i][j]` vary. Now you
   must send a *count* along with the data, and the receiver must know where
   each incoming block lands. This is exactly lesson 16's setup.
2. **Pairwise exchange schedule.** Instead of issuing all `n²` copies at once
   (which oversubscribes links), exchange in `log₂n` rounds: round `k`, rank
   `r` swaps with `r ^ (1<<k)`. Measure — it should saturate links better at
   large `n`.
3. **Self-block optimization.** The `src==dst` block doesn't need to move —
   it's already in the right place. We skip it (a `cudaMemcpy` within the same
   device would be wasted HBM bandwidth). Confirm the diagonal is unchanged.
4. **Compare to NCCL.** `ncclAllToAll` (if available) uses the halving-doubling
   schedule and is much faster at large `n`.

---

# Performance Analysis

- **Naive** issues `(n-1)` sends and `(n-1)` receives per rank. On an NVSwitch
  fabric all of these can proceed in parallel (every pair has a dedicated
  link), so wall-clock ≈ `S/n / BW` per link — already near-optimal for
  full-bisection fabrics.
- **On a mesh (no NVSwitch)**, multiple pairs share links and serialize. Then
  the halving-doubling schedule (round `k` pairs up ranks differing in bit
  `k`) is better — it bounds the contention.
- **The hard part of MoE dispatch isn't the AllToAll itself** — it's that the
  block sizes are unknown until routing runs (lesson 15), and variable, so
  you can't pre-schedule clean rounds. DeepEP's contribution is doing this
  *with low latency* and *with overlap* (lessons 13, 17, 19).

---

# Exercises

1. **Halving-doubling.** Implement the `log₂n`-round schedule. For n=8 it
   should beat naive on a mesh.
2. **Non-uniform AllToAll.** Add a `counts[n][n]` matrix; rank `i` sends
   `counts[i][j]` tokens to rank `j`. The receiver must concatenate. You've
   just written the skeleton of MoE dispatch — lesson 16 fills in routing.
3. **Async overlap.** Issue all sends on one stream and all receives
   (implicit) on another; use events to express "receiver may read slot `i`
   only after rank `i`'s send completes." (Foreshadows NVSHMEM signaling,
   lesson 14.)

---

# DeepEP Connection

```
Lesson 08  AllToAll (uniform block sizes)
   ↓
Lesson 16  mini MoE dispatch (non-uniform AllToAll, fixed routing)
Lesson 17  mini DeepEP     (non-uniform AllToAll, low-latency, channel-parallel)
   ↓
DeepEP     intranode_dispatch / internode_dispatch
```

DeepEP's dispatch **is** an AllToAll — specifically a **non-uniform,
routing-driven AllToAll** where:

- the block `(i,j)` size is `num_tokens_routed[i][j] · hidden_dim`,
- the *contents* of block `(i,j)` are the tokens on GPU i whose top-k experts
  live on GPU j (gathered by an index kernel, lesson 15),
- completion is signaled per-block via flags (lessons 13–14) so the receiver
  can start expert compute as soon as its block arrives, without a global
  barrier.

Today's uniform AllToAll is the substrate; the next nine lessons add routing,
variable sizes, signaling, and overlap — until you arrive at DeepEP.
