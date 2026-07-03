# Lesson 16 — Mini MoE Dispatch

> Wire lessons 8 and 15 together. Route tokens (lesson 15), then dispatch them
> to the right GPU using a **non-uniform AllToAll** (lesson 8) driven by the
> count matrix. This is a MoE dispatch layer, minus only the low-latency
> tricks (those are lesson 17).

---

# Overview

## What are we building?

A complete (if small) **MoE dispatch** across `n` GPUs:

1. Each GPU holds `T/n` tokens.
2. Run the gate + top-1 + count (lesson 15) **on each GPU** for its local
   tokens, then **AllReduce the count matrix** so every GPU knows the global
   `count[src][dst]`.
3. Compute prefix-sum offsets (lesson 15).
4. **Gather** each GPU's outgoing tokens into per-destination buckets
   (contiguous, using the offsets).
5. **AllToAll** (lesson 8): each GPU sends its `dst`-bucket to GPU `dst`.
6. GPU `dst` receives its tokens into a contiguous receive buffer indexed by
   the column-prefix-sum offsets.

```
   GPU0 [tokens] ──gate──▶ assign ──count──▶ [count 0,*] ─┐
   GPU1 [tokens] ──gate──▶ assign ──count──▶ [count 1,*] ─┤  AllReduce
   ...                                                    ├─▶ global count[n][n]
                                                          │  -> offsets[n][n]
   GPU0 gather bucket dst=1 ──AllToAll──▶ GPU1 recv[offset0,1 .. +count0,1]
```

## Why does it matter?

This is the **structure of a real MoE layer's dispatch**, built from primitives
you already wrote. The only things missing vs production (DeepEP) are:

- NVSHMEM device-side puts instead of host `cudaMemcpyPeerAsync` (lesson 17),
- multi-channel overlap (lesson 17),
- FP8 quantization (lesson 19),
- the combine (reverse) pass.

Once you understand this lesson, DeepEP is "the same thing, faster."

## Where is it used in LLM inference?

- **Every MoE layer's dispatch** (DeepSeek-V3, Mixtral, etc.).
- After this dispatch, each GPU runs its local experts on the tokens it
  received, then **combines** (the reverse AllToAll) — which we sketch in the
  exercises.

---

# Goal

- Compose routing (lesson 15) + AllToAll (lesson 8) into a working dispatch.
- See the **non-uniform** sizes: each `(src,dst)` transfer is
  `count[src][dst] · hidden_dim` bytes, different for every pair.
- Verify tokens land at the right destination offset.

---

# Background

## Why AllReduce the count matrix

Each GPU only routes its *own* tokens, so it only knows its row of `count`.
But to receive, GPU `dst` needs to know the **total** tokens incoming
(`Σ_src count[src][dst]`) and the per-source offsets. So we AllReduce (sum)
the `n×n` count matrix — every GPU gets the global plan. This is a tiny
AllReduce (64 ints for n=8) but it's a real collective, and it's on the
critical path of every MoE layer.

## Gather before send

Tokens destined for different experts are interleaved in the input. Before
sending, each GPU **gathers** them into contiguous per-`dst` buckets so the
AllToAll sends are contiguous runs. The gather uses the per-row prefix sum
of `count[src][*]` (different from lesson 15's column prefix — careful!).

| Prefix direction | Use |
|------------------|-----|
| row prefix of `count[src][*]` | **send** gather offsets on src (where to write each bucket locally) |
| column prefix of `count[*][dst]` | **recv** offsets on dst (where each src's bucket lands) |

---

# Architecture Diagram

```
   Per GPU (src):
     tokens[T/n] ──gate──▶ assign[T/n] ──▶ count[src][*] (row)
                                                │  + row prefix -> send offsets
                                                ▼
                                          gather into sendbuf[dst buckets]
                                                │
                                                ▼  AllToAll (lesson 8)
   Per GPU (dst):                          recvbuf
     global count[*][dst] (column) ──▶ col prefix -> recv offsets
     tokens land at recvbuf[ offset[src][dst] .. +count[src][dst] ]
```

---

# Source Code Walkthrough

`moe_dispatch.cu`:

- Reuses `gate_kernel`, `top1_kernel`, `count_kernel` from lesson 15.
- `allreduce_count(count, n)` — naive sum-across-GPUs of the `n×n` matrix
  (each GPU hosts its row; we peer-copy rows to GPU 0, sum, broadcast back).
  Small enough that naive is fine.
- `gather_kernel(assign, tokens, sendbuf, row_prefix, T_local, n, E, D)` —
  for each local token, atomic-add into its `dst`-bucket at the right offset.
  Produces contiguous per-dst buckets in `sendbuf`.
- `dispatch_alltoall(sendbuf, recvbuf, count, offsets, ...)` — for each
  `(src,dst)`, peer-copy `count[src][dst]·D` floats.
- `main()` — runs the full pipeline on `n` GPUs and verifies that GPU `dst`
  received exactly the tokens whose assigned expert lives on `dst`.

(See the file. The gather kernel uses an atomic counter per bucket to assign
each token a unique slot — the standard "scatter by atomic" pattern.)

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target moe_dispatch
```

---

# Run

```bash
./build/lesson16-moe-dispatch/moe_dispatch
# T E D
./build/lesson16-moe-dispatch/moe_dispatch 2048 8 256
```

---

# Expected Output

```
==== lesson 16: mini MoE dispatch ====
n=4 GPUs, T=2048 tokens, E=8 experts, D=256 hidden, top-1

[after routing + allreduce count]
global count[src][dst]:
       dst0  dst1  dst2  dst3
src0    257   245   263   251
src1    248   260   247   253
...

[after gather + AllToAll]
GPU0 received 1010 tokens (experts 0,1 live here)
GPU1 received 1024 tokens
GPU2 received 1011 tokens
GPU3 received 1003 tokens
all tokens routed to the correct GPU: YES
```

The "all tokens routed to the correct GPU: YES" line is the proof: every
token ended up on the GPU hosting its assigned expert.

---

# Experiment

1. **Skew the gate.** Make expert 0 dominant. `count[*][dst0]` grows; GPU 0
   receives more tokens. The dispatch still works — variable sizes handled.
2. **Add the combine.** After "running experts" (fake it: multiply received
   tokens by their expert id), reverse the AllToAll to send outputs back to
   the originating GPUs. Verify each token's output lands at its original
   position. You've now built a full (if tiny) MoE layer.
3. **Top-2.** Each token dispatches to 2 experts. The count matrix now sums
   to `2T`. The combine must weight the two expert outputs by gate softmax
   scores and sum them.
4. **Capacity.** Cap `count[src][dst]` at `capacity`; drop excess tokens.
   Measure the drop rate under skew. (DeepEP avoids this entirely.)

---

# Performance Analysis

- **Routing** (gate + top1 + count) is compute, runs at GEMM/scan speed. Not
  the bottleneck.
- **Count AllReduce** is tiny but on the critical path: every MoE layer pays
  one `n²`-int AllReduce before it can dispatch. At `n=8` it's negligible; at
  `n=64` and low latency it matters.
- **Gather** is memory-bound: each token's `D` floats are written once
  atomically. The atomic is the cost — at high `D` it's hidden behind the
  memory traffic; at low `D` it dominates.
- **AllToAll** is the bulk transfer. With host `cudaMemcpyPeerAsync` we pay
  ~µs launch overhead per `(src,dst)` pair — `n²` of them. That's the cost
  lesson 17 eliminates with NVSHMEM device-side puts.

---

# Exercises

1. **Implement combine.** Reverse the dispatch: each expert GPU sends outputs
   back using the *transposed* count matrix. Verify round-trip correctness.
2. **Weighted combine.** Store gate weights in the dispatch buffer alongside
   tokens; combine multiplies expert output by weight before scattering back.
3. **Fuse gather + send.** Instead of gather-then-send, have one kernel per
   `dst` that streams tokens directly into the peer copy (or NVSHMEM put).
   Saves the `sendbuf` write. (DeepEP does this.)

---

# DeepEP Connection

```
Lesson 15  routing -> count matrix
Lesson 16  + gather + AllToAll dispatch  (host-orchestrated)    <- you are here
Lesson 17  + NVSHMEM device-side puts + channels (low latency)
   ↓
DeepEP     intranode_dispatch:
             - routing produces count matrix (lesson 15)
             - count is AllReduced across GPUs (today)
             - gather + NVSHMEM put into symmetric recv buffers (lesson 17)
             - per-channel flag signaling (lesson 14)
             - FP8 quantization option (lesson 19)
```

DeepEP's dispatch is, structurally, **today's pipeline** — the differences are
all in *how* the AllToAll is executed (NVSHMEM + channels + quant) and *how*
completion is signaled (flags, not barriers). Lesson 17 makes those upgrades.
