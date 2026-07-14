# Lesson 17 — Mini DeepEP Dispatch

> The capstone. Take lesson 16's MoE dispatch and re-cast it the DeepEP way:
> **device-side NVSHMEM puts** into symmetric channel buffers, **per-channel
> flag signaling**, and **multiple channels in parallel** to hide latency. This
> is a faithful (if tiny) recreation of DeepEP's `intranode_dispatch` shape.

---

# Overview

## What are we building?

A low-latency MoE dispatch that mirrors DeepEP's architecture:

1. **Routing** (lesson 15) produces the count matrix; AllReduce it (lesson 16).
2. Each GPU **gathers** its outgoing tokens per destination (lesson 16).
3. Instead of host `cudaMemcpyPeerAsync`, a **dispatch kernel** on each GPU
   issues `nvshmem_putmem` directly into each destination GPU's symmetric
   receive buffer — one put per `(src, dst, chunk)`.
4. After each batch's puts, `nvshmem_quiet` + a **flag put** signals "tokens
   ready" to the destination (lesson 14).
5. **Multiple channels** (separate symmetric buffers + flags) run in parallel
   on separate streams, so while channel A waits for `quiet`, channels B/C/D
   stream (lesson 10/11 overlap).

```
   DeepEP-shape dispatch (one src -> all dsts, N channels):

   src GPU:
     for each channel c:
       for each dst d (on stream c):
         gather chunk of tokens for d
         nvshmem_putmem(recvbuf_d[c], chunk, d)   // device-initiated
         nvshmem_quiet()
         nvshmem_int_put(ready_flag_d[c], seq, d) // signal
   dst GPU:
     for each channel c:
       poll ready_flag[c] == seq
       (expert kernel may now read recvbuf[c])
```

## Why does it matter?

This lesson *is* DeepEP, minus production tuning. Every design choice maps:

| Today                         | DeepEP                          |
|-------------------------------|---------------------------------|
| `nvshmem_putmem` per chunk    | `intranode_dispatch` core write |
| `nvshmem_quiet` + flag put    | per-batch `notify`              |
| N channels on N streams       | `num_channels` parameter        |
| symmetric `recvbuf`           | symmetric `dispatch_buffer`     |
| count matrix + offsets        | `num_tokens_received` + prefix  |
| (skip) FP8 quantization       | `dispatch_fp8` path             |

After this, lesson 18's source map will feel familiar — you've built the
skeleton.

## Where is it used in LLM inference?

This *is* the MoE dispatch used in DeepSeek-V3-class inference (and training).
The "low-latency" variant (small batch, few channels, tight polling) serves
**prefill/decode latency-critical paths**; the "normal" variant (many channels,
high throughput) serves **batched decode**.

---

# Goal

- Implement dispatch with device-initiated NVSHMEM puts (no host copies on the
  data path).
- Add per-channel ready flags with `quiet`-guarded puts.
- Run multiple channels in parallel and see the latency improve vs single-channel.
- Match the result of lesson 16 (tokens land on the right GPU) but faster.

---

# Background

## Why device-side puts beat host copies

Lesson 16's AllToAll issued `n²` `cudaMemcpyPeerAsync` calls from the host —
~µs of launch overhead each, serialized on the host thread. A single dispatch
kernel issues all `n²` (× chunks) puts from the GPU, each at ~100 ns, and the
host enqueues *one* kernel. For small tokens (the latency-critical regime),
this is the difference between ~50 µs and ~5 µs dispatch.

## Why channels

One `nvshmem_quiet` per dst serializes that dst's puts. With N channels, you
issue dst 0's batch on channel 0, dst 1's on channel 1, …, and the `quiet`s
overlap across channels. More channels = more parallelism = lower wall-clock —
up to the point where you saturate NVLink or run out of streams.

## The signaling discipline

```
producer (src), per channel c, per dst d:
    nvshmem_putmem(dst_recvbuf[c][d], tokens, bytes, d)
    nvshmem_quiet()                                   // data landed
    int seq = batch + 1;
    nvshmem_int_put(dst_ready[c][d], &seq, 1, d)      // tell dst "ready"
    nvshmem_quiet()

consumer (dst), per channel c:
    while (*ready[c] != expected_seq) ;                // poll local flag
    // tokens are in recvbuf[c]; expert kernel may run
```

The double-`quiet` (after data, after flag) is the contract. Skip either and
you race.

---

# Architecture Diagram

```
        src GPU (PE s)                              dst GPUs (PE d)
   ┌──────────────────────┐                  ┌──────────────────────┐
   │ dispatch kernel:     │  putmem chan0    │ recvbuf[0] (symm)    │
   │  for c in channels:  │ ───────────────▶ │ ready[0]   (symm) ◀──┐
   │   for d in dsts:     │  putmem chan1    │ recvbuf[1]           │ │ poll
   │     gather chunk     │ ───────────────▶ │ ready[1]             │ │
   │     putmem -> d      │  ...             │ ...                  │ │
   │     quiet            │                  │                      │ │
   │     put ready flag   │ ─── flags ─────▶ │ (expert kernel waits │ │
   └──────────────────────┘                  │  on ready, then runs)│ │
                                             └──────────────────────┘ │
                                                 ▲                     │
                                                 └─ each channel polled independently
```

---

# Source Code Walkthrough

`mini_deepep.cu`:

- `dispatch_kernel(tokens, assign, count, send_offsets, recvbufs, ready_flags, ...)` —
  one block per `(channel, dst)`. Each block gathers its slice of tokens for
  that dst, `nvshmem_putmem`s them into the dst's symmetric `recvbuf`, `quiet`s,
  and puts the ready flag. Issuing all `(channel, dst)` blocks at once gives
  the parallelism.
- `consumer_kernel` — on each dst, polls each channel's ready flag and (in
  this demo) records that the channel's tokens arrived, so we can verify.
- `main()` — runs routing + count (lessons 15–16), then the dispatch kernel
  on each PE, syncs, verifies.

(The code is intentionally compact — ~150 LOC of kernels. Read it alongside
the diagrams above.)

---

# Build

```bash
cmake -S . -B build -DBUILD_NVSHMEM_LESSONS=ON -DNVSHMEM_DIR=/path/to/nvshmem
cmake --build build -j --target mini_deepep
```

---

# Run

```bash
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/nvshmem_12/nvshmrun -np 2 \
    ./build/lesson17-mini-deepep/mini_deepep
# T E D channels
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/bin/nvshmem_12/nvshmrun -np 4 \
    ./build/lesson17-mini-deepep/mini_deepep 2048 8 256 4
```

---

# Expected Output

```
==== lesson 17: mini DeepEP dispatch ====
n=4 PEs, T=2048, E=8, D=256, channels=4

[routing + count AllReduce done]
global count[src][dst]:
       dst0  dst1  dst2  dst3
src0    257   245   263   251
...

[dispatch via NVSHMEM puts, 4 channels]
PE0 dispatched 1016 tokens
PE1 dispatched 1024 tokens
PE2 dispatched 1011 tokens
PE3 dispatched 1003 tokens
all tokens arrived at correct PE: YES
dispatch latency: 0.31 ms   (vs lesson 16 host-orchestrated ~1.4 ms)
```

(Numbers illustrative. The win is the latency drop vs lesson 16, especially
for small T.)

---

# Experiment

1. **Channel sweep.** Run with channels = 1, 2, 4, 8. Latency should drop,
   then plateau (NVLink saturation or stream-slot limit). Find the knee.
2. **Small T (latency regime).** Set T=64. Lesson 16's host overhead dominates
   (~µs per copy × n²); today's device puts stay flat. The gap is largest
   here — this is the regime DeepEP's "low-latency" kernel targets.
3. **Large T (throughput regime).** Set T=65536. Both approaches approach
   NVLink bandwidth; the device-put advantage shrinks. DeepEP's "normal"
   kernel targets this.
4. **Drop a quiet.** Remove the `nvshmem_quiet` after the data put. The ready
   flag may arrive before the data; the consumer reads garbage. Reproduce.
5. **Add FP8.** Quantize each token to FP8 before the put (4 bytes → 1 byte +
   scale). Bandwidth drops ~4×; the receiver dequantizes. This is DeepEP's
   `dispatch_fp8` path (lesson 19).

---

# Performance Analysis

- **Latency** ≈ `n_dst · (putmem + quiet + flag_put + quiet) / channels`. The
  `quiet`s are the fixed cost; channels parallelize them. At small T, this
  floor (~µs) is the whole story.
- **Throughput** at large T ≈ `total_bytes / NVLink_BW`. Channels stop helping
  once NVLink saturates.
- **The host is uninvolved in the data path.** It launches one dispatch
  kernel per PE and waits. That's the structural reason this beats lesson 16:
  no per-copy launch overhead, no host serialization.
- **Comparison to real DeepEP:** real DeepEP additionally (a) fuses the gather
  into the dispatch kernel (no separate `sendbuf`), (b) uses FP8 for
  bandwidth, (c) tunes channel count and chunk size per GPU topology, (d)
  overlaps dispatch with the *previous* layer's compute via events. Today's
  version has the skeleton; lesson 19 adds the meat.

---

# Exercises

1. **Fuse gather + put.** Have the dispatch kernel read tokens directly from
   the input array by assignment index, putting as it goes — no `sendbuf`.
   Saves one HBM write per token. (DeepEP does this.)
2. **Implement combine.** Reverse: each expert PE puts outputs back to the
  origin PE using the transposed count matrix and a second set of flags.
   Verify round-trip.
3. **Tune chunk size.** Split each `(src,dst)` bucket into K chunks, one per
   channel-round, so a large bucket parallelizes across channels. Find the K
   that minimizes latency at T=8192.
4. **Overlap with expert compute.** Use events (lesson 11) so expert e's
   kernel starts the moment channel e's ready flag fires, while other
   channels are still streaming. This is the DeepEP dispatch/compute overlap.

---

# DeepEP Connection

```
Lesson 17  mini DeepEP dispatch (NVSHMEM + channels + flags)
   ↓
Lesson 18  reading the real DeepEP source  (recognize today's pieces)
Lesson 19  optimizing: FP8, fused gather, tuned channels, overlap
Lesson 20  full architecture writeup
   ↓
DeepEP     intranode_dispatch / internode_dispatch / *_combine
           (production version of exactly this lesson)
```

You have now built, from `cudaMemcpy` upward, a working model of DeepEP's
dispatch. The remaining lessons (18–20) step back: lesson 18 maps today's
pieces onto the real source files, lesson 19 lists the optimizations that
separate this teaching version from production, and lesson 20 is the full
architectural writeup. The hard part — building the mental model — is done.
