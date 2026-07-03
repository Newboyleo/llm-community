# Lesson 19 — Optimizing Communication

> Take lesson 17's mini DeepEP and apply DeepEP's real optimizations one at a
> time, measuring each. This is the playbook that turns a correct dispatch
> into a fast one — and the catalog of knobs you tune when reading DeepEP.

---

# Overview

## What are we doing?

Lesson 17 built a *correct* low-latency dispatch. It left performance on the
table. This lesson walks the optimization ladder, one rung at a time:

1. **Batch the quiet** (one quiet per batch, not per token)
2. **Use `putmem` instead of per-element `float_put`** (one put per token, not D)
3. **FP8 quantization** (4 bytes → 1 byte + scale)
4. **Tune channel count** (find the knee)
5. **Tune chunk size** (latency vs overhead)
6. **Fuse gather into the dispatch kernel** (kill the `sendbuf`)
7. **Overlap dispatch with expert compute** (events + SM split)

Each rung has a measurable effect. By the end you'll know *which knob to turn
for which symptom* — the actual skill of using DeepEP in production.

```
   lesson 17 baseline
        │  + batch quiet            ─▶ fewer quiets, lower overhead
        │  + putmem per token       ─▶ 1 put/token, not D
        │  + FP8                    ─▶ 4× less bandwidth
        │  + tuned channels         ─▶ parallel quiets hide latency
        │  + tuned chunk size       ─▶ balance latency/overhead
        │  + fused gather           ─▶ no sendbuf write
        │  + overlap with compute   ─▶ dispatch hidden behind experts
        ▼
   DeepEP-class throughput/latency
```

## Why does it matter?

A correct MoE dispatch that's 5× slower than DeepEP is useless in production.
The difference is *exactly* the seven rungs above. None of them change the
algorithm — they change how the algorithm is *scheduled onto the hardware*.
This is the lesson where "I built a working DeepEP" becomes "I can make it
fast."

## Where does it fit in LLM inference?

Every DeepEP knob (`num_channels`, `num_max_nvl_chunk`, `fp8` mode, low-latency
vs normal) corresponds to a rung here. Tuning them for your GPU topology and
batch size is the day-to-day job of an inference engineer working with MoE.

---

# Goal

- For each optimization, predict the effect, apply it, measure.
- Understand the **latency vs throughput** tradeoff each knob controls.
- Know which knob to turn when you see a symptom (high latency, low bandwidth,
  SM starvation, NVLink saturation).

---

# Background: the seven rungs

### 1. Batch the quiet
`nvshmem_quiet` is expensive (~hundred ns + a memory-system flush). Lesson 17
already puts one token per `(thread, channel)` then quiets — but if you put
*B* tokens before the quiet, you pay the quiet once per B. **Effect: linear
throughput gain in B until NVLink saturates.**

### 2. `putmem` vs per-element puts
Lesson 17 does `nvshmem_float_put` D times per token (D = hidden dim). That's
D puts × D quiet-overheads-amortized-but-still-D-API-calls. `nvshmem_putmem`
moves all D floats in one call. **Effect: ~D× fewer NVSHMEM API calls per
token.**

### 3. FP8 quantization
Token hidden states are typically BF16 (2 bytes). FP8 (1 byte + a per-block
scale) halves the bandwidth, with a small dequant cost on the receiver.
DeepEP's `dispatch_fp8` path does this. **Effect: ~2× bandwidth gain, small
accuracy cost (FP8 has ~2-3 bits of mantissa).**

### 4. Channel count
More channels = more parallel quiets = lower wall-clock per batch — *until*
you saturate NVLink or exhaust stream slots. **Effect: latency drops, then
plateaus; the plateau is the hardware limit.**

### 5. Chunk size
Splitting a `(src,dst)` bucket into chunks lets multiple channels cooperate
on one bucket. Small chunks → low latency, high overhead. Large chunks → high
throughput, less parallelism. **Effect: tunes the latency/throughput knee.**
This is DeepEP's `num_max_nvl_chunk` / `num_rdma_chunk`.

### 6. Fuse the gather
Lesson 16 writes tokens into a `sendbuf`, then the dispatch reads `sendbuf`
and puts. That's an extra HBM round-trip. Fusing — reading tokens directly
from the input by assignment index and putting as you go — removes the
`sendbuf` write. **Effect: ~½ the HBM traffic of the dispatch stage.**

### 7. Overlap with compute
Run expert `e`'s kernel on a separate stream the moment channel `e`'s ready
flag fires, while other channels are still streaming. Requires SM partitioning
(reserve SMs for expert compute so dispatch doesn't starve). **Effect: hides
dispatch latency behind compute — the throughput win for large batches.**

---

# Architecture Diagram

```
   Optimization ladder (each layer builds on the one below):

   ┌─────────────────────────────────────────────────────┐
   │ 7. overlap dispatch ‖ expert compute  (events+SMs)  │  ← throughput
   │ 6. fused gather (no sendbuf)                        │
   │ 5. chunk-size tuning                                │
   │ 4. channel-count tuning                             │
   │ 3. FP8 quantization                                 │  ← bandwidth
   │ 2. putmem per token (not per float)                 │
   │ 1. batched quiet                                    │  ← overhead
   └─────────────────────────────────────────────────────┘
                  lesson 17 baseline (correct, slow)
```

---

# Source Code Walkthrough

This lesson ships a **single parameterized kernel** with compile-time /
runtime switches for each optimization, plus a sweep harness that measures
each combination. See `optimizing.cu`:

- `DISPATCH_OPT` struct: `{ bool putmem; bool fp8; int channels; int chunk; bool fused; bool overlap; }`.
- `dispatch_kernel<OPT>` — templatized on the optimization flags; the
  compiler eliminates the dead branches per configuration.
- `sweep()` — runs the kernel across a matrix of OPT settings and prints a
  table: latency, bandwidth, SM utilization proxy.

(For space, the file implements rungs 1–4 fully and stubs 5–7 with hooks
marked `// EXERCISE`. Filling them in is the exercise set.)

Key shape (batched quiet + putmem):

```c
// gather B tokens for this (ch, dst) into shared mem, then ONE putmem + quiet
for (int b = 0; b < B; ++b) { scratch[b*D + d] = ...; }
nvshmem_putmem(&recvbuf[slot*D], scratch, B*D*sizeof(float), dst);
nvshmem_quiet();            // ONE quiet per B tokens  (rung 1)
// (rung 2: putmem, not D float_puts)
```

---

# Build

```bash
cmake -S . -B build -DBUILD_NVSHMEM_LESSONS=ON -DNVSHMEM_DIR=/path/to/nvshmem
cmake --build build -j --target optimizing
```

---

# Run

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/lesson19-optimizing/optimizing
```

---

# Expected Output

```
==== lesson 19: optimizing communication ====
baseline (lesson-17 shape): per-float put, 1 channel, no batch
  T=2048  latency 2.41 ms   bw 84 GB/s

rung 1: batched quiet (B=32)
  latency 0.88 ms   bw 230 GB/s     (2.7× — fewer quiets)

rung 2: putmem per token
  latency 0.41 ms   bw 494 GB/s     (2.1× — fewer API calls)

rung 3: FP8
  latency 0.27 ms   bw 750 GB/s     (1.5× — half the bytes; dequant cost shows)

rung 4: channels=8 (was 1)
  latency 0.14 ms   bw 1440 GB/s    (1.9× — parallel quiets; near NVLink sat)

rung 5: chunk=4KiB  (exercise)
rung 6: fused gather (exercise)
rung 7: overlap with compute (exercise)
```

(Numbers illustrative; the *ratios* are the point. Notice diminishing returns
at the top — rungs 1–2 give the biggest wins, which is why they're the first
things DeepEP does.)

---

# Experiment

1. **Run the sweep.** The harness varies `channels ∈ {1,2,4,8,16}` and `B ∈
   {1,8,32,128}`. Plot latency vs channels for each B. Find the knee (where
   more channels stops helping).
2. **FP8 accuracy.** Add a dequant step and compare the dispatched tokens to
   the BF16 baseline. Compute the max relative error. Decide if it's
   acceptable for your model.
3. **Implement rung 5** (chunk size). Split each `(src,dst)` bucket into K
   chunks assigned round-robin to channels. Find the K that minimizes latency
   at T=8192.
4. **Implement rung 6** (fused gather). Read tokens directly from the input
   by `assign[t]` and put, skipping `sendbuf`. Measure the HBM-traffic
   reduction via `nsys`.
5. **Implement rung 7** (overlap). Launch a dummy "expert" kernel on a second
   stream, gated by an event from each channel's ready flag. Measure how much
   of the dispatch hides behind it.

---

# Performance Analysis

| Rung              | What it beats            | Diminishing returns?         |
|-------------------|--------------------------|------------------------------|
| 1 batch quiet     | quiet overhead           | saturates at small B         |
| 2 putmem          | API call overhead        | one-time ~D× gain            |
| 3 FP8             | bandwidth                | limited by accuracy budget   |
| 4 channels        | quiet serialization      | plateaus at NVLink sat       |
| 5 chunk size      | latency/overhead knee    | problem-dependent            |
| 6 fused gather    | HBM traffic              | one-time ~2× of dispatch HBM |
| 7 overlap         | hides dispatch latency   | bounded by compute/dispatch ratio |

The **first three rungs** are where most of the win lives for small batches
(latency regime). Rungs 4–7 matter most for large batches (throughput
regime). DeepEP's **low-latency** mode leans on 1–3 + 4 with few channels;
its **normal** mode leans on 4–7.

---

# Exercises

1. **Implement each rung** as a separately compilable variant and benchmark.
   This is the most valuable exercise in the whole lab — it's literally the
   job.
2. **Find the bottleneck.** At the top of the ladder, profile with Nsight
   Compute / Systems. Is the remaining time in NVLink, in HBM, in the quiet,
   or in launch? The answer tells you what *would* help next (and at this
   point, the answer is usually "nothing — you're at the hardware ceiling").
3. **Compare to real DeepEP.** Run DeepEP's own benchmark on the same
   hardware. The gap between your optimized kernel and DeepEP is the sum of
   tuning details (chunk schedules, SM-split ratios, register pressure
   tuning) that are beyond a teaching lab — but each is a known, named
   technique you can now recognize.

---

# DeepEP Connection

```
Lesson 19  optimization ladder (rungs 1-7)
   ↓
DeepEP     every knob in the API is one of these rungs:
             num_channels            = rung 4
             num_max_nvl_chunk       = rung 5
             num_rdma_chunk          = rung 5 (internode)
             fp8 mode                = rung 3
             "fused" gather          = rung 6 (always on in DeepEP)
             SM split / overlap      = rung 7
             batched quiet           = rung 1 (always on)
             putmem                  = rung 2 (always on)
```

When you tune DeepEP in production, you are turning the rung-4 and rung-5
knobs (channels and chunks) for your topology, deciding rung-3 (FP8) based on
your accuracy budget, and relying on DeepEP to have already baked in rungs 1,
2, 6, 7. This lesson is the map from "knob name" to "which rung, which
effect."

Next: lesson 20 synthesizes everything into a single architectural writeup.
