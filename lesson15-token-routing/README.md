# Lesson 15 — Token Routing

> The "brain" of MoE. Each token runs through a **gate**, gets a score per
> expert, picks its top-k experts, and we end up with a **per-(src,dst) token
> count matrix** — the routing plan that drives dispatch.

---

# Overview

## What are we building?

A GPU kernel that simulates the routing stage of a MoE layer:

1. **Gate logits:** for each token `t`, compute `logits[t] = x[t] · W_gate`
   (E logits, one per expert).
2. **Top-k:** for each token, pick the k experts with the highest logits.
   (We use k=1 for simplicity; real MoE uses k=2 or k>2.)
3. **Count matrix:** `count[src][dst]` = number of tokens on GPU `src` whose
   chosen expert is on GPU `dst`. This is the **dispatch plan**.
4. **Prefix sums:** per `src`, a prefix sum over `dst` gives the write offset
   for each destination bucket — the index used by the dispatch gather kernel.

```
   tokens ──▶ gate (matmul) ──▶ logits [T x E]
                                  │ argmax/topk
                                  ▼
                              assignment [T]   (expert id per token)
                                  │ count by (src_pe, dst_pe)
                                  ▼
                              count [n x n]    <- the dispatch plan
                                  │ prefix sum per row
                                  ▼
                              offsets [n x n]  <- where each token goes
```

## Why does it matter?

Routing is **what makes MoE dispatch non-uniform**. In lesson 8's AllToAll,
every `(src,dst)` block was the same size. In MoE, `count[src][dst]` varies
with the input — some experts are popular, some are idle. The dispatch must
respect these variable sizes, and the *combine* must invert them. Everything
DeepEP optimizes (variable-size buffers, prefix-sum offsets, channel sizing)
starts from this count matrix.

## Where is it used in LLM inference?

- **Every MoE layer** routes tokens to experts before dispatch.
- **Expert load balancing loss** is computed from the count matrix (we don't
  train here, but the matrix is the same).
- **Capacity planning:** `max(count[src][dst])` over a batch sets the buffer
  size needed (or, in DeepEP's case, you skip capacity and use exact counts).

---

# Goal

- Implement gate matmul + top-1 + count + prefix-sum on the GPU.
- Understand the count matrix as the **routing plan** for a non-uniform
  AllToAll.
- See why the prefix sum is needed: the dispatch kernel needs to know, for
  each token going to `dst`, *at which offset* in `dst`'s receive buffer it
  should land.

---

# Background

## Gate and top-k

```
x[t] : hidden_dim vector (the token's hidden state)
W_gate : hidden_dim x E matrix
logits[t, e] = x[t] · W_gate[:, e]
assign[t] = argmax_e  logits[t, e]      (top-1)
```

For top-k (k>1), you'd keep the k highest; each token then duplicates into k
"virtual tokens," one per chosen expert. We do top-1 for clarity.

## From assignment to count matrix

Each token `t` lives on some GPU `src = t / (T/n)`. Its chosen expert `e`
lives on some GPU `dst = e / (E/n)`. So:

```
count[src][dst] += 1   for each token t with assign[t]=e
```

`count` is `n x n`. Row `src` tells you, for the tokens on GPU `src`, how
many go to each destination GPU.

## Prefix sum → offsets

To dispatch, GPU `src` needs to write its tokens destined for `dst` into a
contiguous run in `dst`'s receive buffer. The **offset** where row `src`'s
`dst`-bucket starts is the prefix sum of `count[*][dst]` over `src`:

```
offset[src][dst] = sum_{s<src} count[s][dst]
```

Now token `t` (on src, going to dst) writes to `recv[dst][ offset[src][dst] + local_index ]`.

This prefix-sum-of-counts is exactly the "prefix matrix" DeepEP computes to
drive its dispatch indexing.

---

# Architecture Diagram

```
   Per-token:   x[t] ──gate──▶ logits[t, 0..E-1] ──top1──▶ assign[t] = e
                                                                 │
   Per-(src,dst):  src = t/(T/n),  dst = e/(E/n)                 ▼
                  count[src][dst] ++                          assign[]
                                                                     │
   Per-row prefix:  offset[src][dst] = Σ_{s<src} count[s][dst]       ▼
                                                              count[n][n] ──▶ offsets[n][n]
```

---

# Source Code Walkthrough

`token_routing.cu`:

- `gate_kernel(x, W, logits, T, E, D)` — plain matmul: `logits[t,e] = Σ_d x[t,d]*W[d,e]`.
- `top1_kernel(logits, assign, T, E)` — for each token, the thread scans its E
  logits and keeps the argmax.
- `count_kernel(assign, count, T, n)` — each token atomically increments
  `count[src][dst]`. (Atomic because many tokens map to the same `(src,dst)`.)
- `prefix_kernel(count, offsets, n)` — per-column prefix sum to get offsets.
  (Column prefix because `offset[src][dst]` accumulates over `src` for a fixed
  `dst`.)
- `main()` — runs all four, prints the count matrix and offsets.

Key lines (count, the heart of it):

```c
__global__ void count_kernel(const int* assign, int* count, int T, int n, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int e = assign[t];
    int src = (t * n) / T;        // which GPU token t lives on
    int dst = e / (E / n);        // which GPU expert e lives on
    atomicAdd(&count[src * n + dst], 1);
}
```

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target token_routing
```

(No NVSHMEM needed — routing is a single-GPU compute stage. We'll feed its
output into the cross-GPU dispatch in lesson 16.)

---

# Run

```bash
./build/lesson15-token-routing/token_routing
# T E D n seed
./build/lesson15-token-routing/token_routing 1024 8 256 4 1234
```

---

# Expected Output

```
==== lesson 15: token routing ====
T=1024 tokens, E=8 experts, D=256 hidden, n=4 GPUs (E/n=2 experts/GPU), seed=1234

assign[0..7] = [5, 5, 3, 3, 6, 6, 4, 4, ...]

count[src][dst]  (rows=src GPU, cols=dst GPU):
       dst0  dst1  dst2  dst3
src0     59    72    72    53
src1     58    69    68    61
src2     55    68    78    55
src3     55    81    63    57

offsets[src][dst] = prefix sum of count[*][dst] over src:
       dst0  dst1  dst2  dst3
src0      0     0     0     0
src1     59    72    72    53
src2    117   141   140   114
src3    172   209   218   169
```

The count matrix is roughly uniform here (random gate weights → uniform
routing). With a trained gate, you'd see skew — some `count[src][dst]` much
larger than others. That skew is what load-balancing losses fight, and what
DeepEP's variable-size buffers handle.

---

# Experiment

1. **Skew the gate.** Make one expert's gate column larger so it's chosen
   more often. Watch `count[*][dst_hot]` grow. This is the imbalance MoE
   suffers from in practice.
2. **Top-2.** Change `top1` to `top2`: each token picks 2 experts, so it
   appears twice in the dispatch (once per expert). `count` now sums to `2T`.
   Real MoE.
3. **Capacity.** Set `capacity = max(count[*][dst])` per dst. Tokens beyond
   capacity are dropped (or overflowed). DeepEP's contribution is to **not**
   use a fixed capacity — it dispatches exactly `count[src][dst]` tokens, no
   padding, no drops.
4. **Atomic contention.** At large T, the `atomicAdd` to `count` becomes a
   bottleneck. Replace with a per-block local count + a final reduce. (This
   is the standard GPU histogram trick.)

---

# Performance Analysis

- **Gate matmul** is `T·E·D` FLOPs — a normal GEMM, runs at GEMM bandwidth.
  In production this is fused with the preceding attention/output projection.
- **Top-1** is `T·E` comparisons, one warp per token is plenty.
- **Count** is `T` atomic adds into `n²` bins. At `n=8`, `n²=64` bins — low
  contention. At `n=64` the bin count grows but contention per bin drops.
- **Prefix sum** is `O(n²)` — trivially cheap. The interesting part is that
  it's a **column** prefix sum (over `src`), not a row prefix sum. Get this
  wrong and every token lands at the wrong offset.

---

# Exercises

1. **Top-2 with duplication.** Implement top-2 routing; each token contributes
  to two experts. The dispatch buffer now holds `2T` "virtual tokens." Verify
  the count sums to `2T`.
2. **Weighted routing.** Attach the gate softmax weight to each routed token.
  The combine (lesson 16) will multiply expert outputs by these weights. Store
  them alongside the token in the dispatch buffer.
3. **Histogram without atomics.** Replace the atomic count with a
  per-warp/block local histogram + reduce. Measure the speedup at large T.

---

# DeepEP Connection

```
Lesson 15  gate + top-k + count[src][dst] + prefix offsets
   ↓
Lesson 16  mini MoE dispatch (uses count + offsets to gather & send)
Lesson 17  mini DeepEP     (same, with NVSHMEM + channels + low latency)
   ↓
DeepEP     the "routing" stage computes exactly this count matrix (called the
           "dispatch count" / "num_tokens_received" matrix). Its prefix-sum
           gives the per-source write offsets into each destination's receive
           buffer. DeepEP's kernels then:
             - gather tokens by offset (lesson 16)
             - ship them via NVSHMEM puts into symmetric buffers (lesson 17)
             - signal per-batch readiness (lesson 14)
```

The count matrix is the **single most important data structure** in DeepEP.
When you read `dispatch`'s indexing code in lesson 18, every
`offset[src][dst]` you see is today's prefix sum. The whole library exists to
move tokens according to this matrix, fast.
