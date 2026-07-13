# Lesson 08 — Generic AllToAllv

> The collective at the heart of MoE dispatch/combine. Rank `src` sends a
> distinct, possibly variable-sized bucket to rank `dst` for every pair
> `(src,dst)`.

---

# Overview

The original AllToAll demo can be pictured as transposing an `n x n` matrix of
equal-sized blocks. That is useful for learning the shape, but real MoE
communication is more general:

- each GPU owns local tokens,
- routing decides which destination GPU hosts each token's expert,
- `count[src][dst]` can differ for every pair,
- dispatch sends token buckets to expert GPUs,
- combine sends expert outputs back to the source GPUs.

This lesson now implements that more general **AllToAllv** shape with plain
`cudaMemcpyPeerAsync`.

```
source GPU r:
  tokens -> bucket by route[t] -> sendbuf[dst buckets]

dispatch:
  sendbuf[src][bucket dst] -> recvbuf[dst][segment src]

expert GPU dst:
  run local "expert" transform on received tokens

combine:
  recvbuf[expert][segment origin] -> returnbuf[origin][segment expert]
  scatter by token id back to original token order
```

---

# What The Code Shows

`alltoall.cu` is organized around an explicit `A2AContext`:

- `counts[src][dst]` — number of tokens moving from source GPU to expert GPU.
- `dispatch_send_offsets` — where source GPU `src` placed destination bucket
  `dst` in its packed send buffer.
- `dispatch_recv_offsets` — where destination GPU `dst` receives source
  segment `src`.
- `combine_*` — the transposed plan used to send expert outputs back.
- per-GPU buffers: `tokens`, `sendbuf`, `recvbuf`, `returnbuf`, `output`.

The main flow is:

1. `build_plan()` creates a deterministic variable-count routing plan.
2. `dispatch()` gathers local tokens into per-destination buckets and calls
   `alltoallv()`.
3. `run_experts()` applies a tiny visible transform on each expert GPU.
4. `combine()` calls `alltoallv()` with the transposed plan and scatters
   results back to original token order.
5. `verify()` checks the full dispatch → expert → combine round trip.

The reusable core is:

```cpp
alltoallv(ctx, srcbufs, dstbufs, counts, send_offsets, recv_offsets);
```

That routine is the generic peer-copy schedule. Dispatch and combine only
differ by which count/offset plan they pass in.

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target alltoall
```

---

# Run

```bash
./build/lesson08-alltoall/alltoall
./build/lesson08-alltoall/alltoall 4096 256
```

Arguments:

- `tokens_per_rank` defaults to `2048`.
- `hidden_dim` defaults to `128` and must be at least `2` because element `0`
  stores a token id for the combine scatter.

---

# Expected Output Shape

```
==== lesson 08: generic alltoallv dispatch/combine ====
n_gpus=4, tokens_per_rank=2048, hidden_dim=128

dispatch count[src][dst] tokens:
       dst0    dst1    dst2    dst3    total
src0   ...
src1   ...

==== dispatch ====
[bw] dispatch remote bytes ...
GPU0 received ... tokens for local experts

==== local expert compute ====
GPU0 expert step processed ... tokens

==== combine ====
[bw] combine remote bytes ...

rank0 output sample:
  token 0   -> expert_gpu ... -> y[0]=0 y[1]=...

round trip result: OK
```

---

# Why This Is Closer To MoE

Uniform AllToAll has equal block sizes, so it looks like a transpose. MoE
dispatch is a routing-driven **non-uniform AllToAllv**:

- block `(src,dst)` contains only tokens from `src` whose expert lives on
  `dst`,
- each block has size `count[src][dst] * hidden_dim`,
- the receiver concatenates incoming source segments,
- combine is the reverse plan plus a scatter/reduce back to token order.

This lesson keeps the transport simple and host-orchestrated. Later lessons
replace pieces with realistic routing, flags, channels, and NVSHMEM-style
device-initiated puts.


                      A2AContext
┌────────────────────────────────────────────────────────────┐
│ Basic Config                                               │
│  n, tokens_per_rank, hidden_dim, total_tokens              │
├────────────────────────────────────────────────────────────┤
│ Dispatch Plan                                              │
│  counts[src][dst]                                          │
│  dispatch_send_offsets[src][dst]                           │
│  dispatch_recv_offsets[src][dst]                           │
├────────────────────────────────────────────────────────────┤
│ Combine Plan                                               │
│  combine_counts (countsᵀ)                                 │
│  combine_send_offsets                                      │
│  combine_recv_offsets                                      │
├────────────────────────────────────────────────────────────┤
│ Routing Metadata                                           │
│  h_route            (token → expert)                       │
│  h_slot_in_bucket   (token 在目标 bucket 中的位置)         │
├────────────────────────────────────────────────────────────┤
│ GPU Buffers (每个 Rank 一份)                               │
│  d_tokens      输入 Token                                  │
│  d_sendbuf     Dispatch 打包后的发送缓冲区                 │
│  d_recvbuf     Expert 接收到的数据                         │
│  d_returnbuf   Combine 返回的数据                          │
│  d_output      最终恢复原顺序的输出                        │
│  d_route / d_slot_in_bucket / d_dispatch_send_base         │
└────────────────────────────────────────────────────────────┘