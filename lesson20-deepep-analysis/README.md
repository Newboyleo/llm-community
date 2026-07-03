# Lesson 20 — Complete DeepEP Analysis

> The capstone writeup. Pull every lesson together into a single architectural
> picture of DeepEP: what it does, why it's structured that way, where each
> piece came from in this lab, and how to reason about it as a system.

---

# Overview

## What is this?

No code. A synthesis: the full DeepEP architecture, explained end-to-end using
only the concepts you built in lessons 1–19. Read this once you've done the
labs; come back to it when you need the big picture.

```
   lessons 1-11   ── the physical & collective primitives
   lessons 12-14  ── NVSHMEM programming model
   lessons 15-17  ── MoE routing + mini DeepEP
   lessons 18-19  ── reading & optimizing the real source
   lesson 20      ── the whole thing, as one system   ← you are here
```

## Why does it matter?

After 19 lessons you understand the *parts*. This lesson is about the
*whole*: how the parts compose, what tradeoffs define the design, and where
the remaining performance ceiling lives. It's the "I now see how this fits
together" moment — the actual goal of the lab.

## Where does it fit in LLM inference?

DeepEP is the communication spine of DeepSeek-V3-class MoE. Understanding it
as a system is what lets you:

- predict its behavior under new hardware / batch sizes,
- diagnose regressions ("dispatch got slower — is it routing, channels, or
  NVLink?"),
- decide whether to tune it, replace it, or write a new one.

---

# Goal

- Explain DeepEP's architecture in one breath and in one page.
- Map every component to its lesson.
- Name the three fundamental tradeoffs that define the design.
- Know where the performance ceiling is and what's beyond it.

---

# DeepEP in one breath

> DeepEP is a **low-latency, high-throughput MoE dispatch/combine library**
> built on **NVSHMEM symmetric heaps** and **device-initiated P2P puts**. It
> consumes a **per-(src,dst) token-count matrix** (computed by the model's
> router), moves tokens to the right expert GPUs via **multi-channel,
> batched, quiet-guarded puts**, signals completion with **per-channel flags**,
> and optionally **quantizes to FP8**. Dispatch and combine are symmetric
> (combine is dispatch with the transposed count matrix and a weighted sum).

That's the whole thing. Everything else is engineering.

---

# Architecture Diagram (the whole system)

```
   ┌──────────────────────────── MODEL (DeepSeek-V3 MoE layer) ────────────────────────────┐
   │                                                                                         │
   │  tokens ──gate──▶ assign ──count──▶ count[src][dst]  ──AllReduce──▶ global count       │
   │    (lesson 15)                          │                          (lesson 6/7)         │
   │                                         ▼                                                │
   │                              prefix offsets  (lesson 15 column-prefix)                 │
   └─────────────────────────────────────────┼──────────────────────────────────────────────┘
                                             │
   ┌─────────────────────────── DeepEP ──────▼──────────────────────────────────────────────┐
   │                                                                                         │
   │  dispatch (intranode/internode)                                                         │
   │    for channel c in 1..N:           (lesson 10/11 streams + events)                     │
   │      for dst d:                                                                          │
   │        gather chunk (lesson 16)  ──or── fused (lesson 19 rung 6)                         │
   │        nvshmem_putmem -> d's symmetric recvbuf   (lesson 12/13)                          │
   │        nvshmem_quiet                                (lesson 14)                          │
   │        flag put "ready" -> d                       (lesson 14)                          │
   │    [optional FP8]                                  (lesson 19 rung 3)                    │
   │                                                                                         │
   │  expert compute (model's kernels, overlapped via events/SM-split — lesson 19 rung 7)    │
   │                                                                                         │
   │  combine (transpose of dispatch)                                                        │
   │    weighted putmem of expert outputs back to origin PEs (lesson 5 reduce + lesson 8)    │
   │                                                                                         │
   └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# Component-by-component, mapped to lessons

| DeepEP component                         | Lesson(s) | What it is                                       |
|------------------------------------------|-----------|--------------------------------------------------|
| NVSHMEM init + symmetric heap            | 12        | the programming model                            |
| Per-channel ring buffer (symmetric)      | 9, 13     | where tokens stream into                         |
| Ready/done flag handshake                | 14        | per-batch completion signaling                   |
| Count matrix + prefix offsets            | 15        | the dispatch plan                                |
| Gather into per-dst buckets              | 16        | make sends contiguous                            |
| AllToAll of variable-size blocks         | 8, 16     | the dispatch itself                              |
| Device-initiated puts (no host)          | 12, 17    | low-latency primitive                            |
| Multi-channel parallelism                | 10, 17    | hide quiet latency                               |
| Combine = transposed dispatch + reduce   | 5, 8, 16  | the reverse pass                                 |
| FP8 quantization                         | 19 rung 3 | bandwidth                                        |
| Batched quiet                            | 19 rung 1 | overhead                                         |
| putmem per token                         | 19 rung 2 | API overhead                                     |
| Channel/chunk tuning                     | 19 rung 4-5| latency/throughput knob                         |
| Fused gather                             | 19 rung 6 | HBM traffic                                      |
| Overlap with expert compute              | 19 rung 7 | throughput                                       |
| Internode = same + RDMA transport        | 17 + RDMA | multi-node                                       |

Every row is something you built.

---

# The three defining tradeoffs

### 1. Latency vs throughput (the mode split)
DeepEP ships **low-latency** and **normal** modes. Low-latency uses **fewer
channels, smaller chunks, tighter polling** — minimum wall-clock for small
batches (prefill, single-token decode). Normal uses **many channels, larger
chunks, overlap** — maximum bandwidth for large batches. You can't have both
in one kernel; the mode switch is the tradeoff made explicit. *(Lessons 17,
19.)*

### 2. Bandwidth vs accuracy (FP8)
FP8 halves the bytes but loses ~2 bits of mantissa. DeepEP makes it optional:
BF16/FP16 for accuracy-critical paths, FP8 when bandwidth-bound. The
quantize/dequant cost is paid on the GPU (cheap, HBM-speed) to save NVLink
(expensive, bottleneck). *(Lesson 19 rung 3.)*

### 3. Overlap vs SM contention
Running dispatch and expert compute simultaneously hides latency but
**contends for SMs**. DeepEP explicitly partitions SMs (some for dispatch
puts, rest for expert kernels) via launch bounds. Too few dispatch SMs →
NVLink underutilized; too few expert SMs → compute starves. The split is
topology- and batch-dependent. *(Lesson 19 rung 7; lesson 10 overlap.)*

---

# The data path, traced end-to-end

Follow one token from `send_x` on PE 0 to the expert kernel on PE 1:

1. **Router** (model code) computes `assign[t] = e`. *(Lesson 15 gate+top1.)*
2. **Count** accumulates `count[0][pe_of(e)] += 1`. *(Lesson 15.)*
3. **AllReduce count** so every PE knows the global matrix. *(Lessons 6/7.)*
4. **Prefix sum** gives `offset[0][pe_of(e)]`. *(Lesson 15 column-prefix.)*
5. **DeepEP dispatch kernel** on PE 0: reads `x[t]`, `nvshmem_putmem`s it
   into PE 1's symmetric `recvbuf` at `offset[0][1]*D`. *(Lessons 12, 17.)*
6. **Quiet + flag put**: PE 0 `nvshmem_quiet`s, then puts `ready[ch][0]=seq`
   to PE 1. *(Lesson 14.)*
7. **PE 1's expert kernel** (on a separate stream) is waiting on that flag's
   event; it fires, reads the token from `recvbuf`, runs the expert FFN.
   *(Lessons 10, 11, 19 rung 7.)*
8. **Combine** (reverse): PE 1 `putmem`s the expert output back to PE 0 at
   the transposed offset, weighted by the gate score. PE 0 sums into the
   token's original position. *(Lessons 5, 8, 16-reverse.)*

Eight steps, each a lesson you've done. DeepEP is the choreography; the steps
are yours.

---

# Where the ceiling is

After all optimizations (lesson 19), the remaining bottlenecks are
**physical**, not algorithmic:

- **NVLink bandwidth** (intranode): ~300–500 GB/s per pair on H100. You can't
  put bytes faster than the wire carries them. Saturating it across all pairs
  simultaneously requires NVSwitch.
- **HBM bandwidth** (the gather/fused-gather + dequant): ~2–3 TB/s per GPU.
  At large D this becomes the limit, not NVLink.
- **Quiet/fence latency floor**: ~hundred ns each. At tiny batches (single
  token), you can't go below a few µs total because of this floor + the poll
  latency. This is why "low-latency" mode exists — to shave the floor.
- **RDMA overhead** (internode): per-message IB overhead (~few µs) sets the
  internode floor, higher than intranode. Hence the larger internode chunks.

Beyond these, you'd need new hardware (fatter NVLink, cheaper fences,
unified HBM/NVLink fabric) — not new algorithms. DeepEP is, for the
algorithms it implements, near the ceiling.

---

# Build / Run

N/A — this is a synthesis lesson. Re-read lessons 17–19 if a component feels
hazy.

---

# Expected Output

You should be able to:

- draw the architecture diagram from memory,
- name the three tradeoffs,
- trace a token through all eight steps,
- point at the ceiling for each regime.

If you can, you've met the lab's final objective.

---

# Experiment (capstone projects)

Pick one to cement the whole lab:

1. **Port mini-DeepEP to your model.** Take lesson 17's kernel, wire it into a
   tiny MoE forward pass (2 experts, 2 GPUs), and run a real — if toy —
   inference. Measure end-to-end latency and identify the bottleneck via
   Nsight Systems.
2. **Implement combine.** You have dispatch (lesson 17); build the symmetric
   combine. Verify a full dispatch→expert→combine round trip reconstructs the
   input (modulo expert transform).
3. **Tune for your box.** Run lesson 19's sweep on your hardware, find the
   optimal `(channels, chunk, FP8?)` for batch sizes {1, 64, 4096}, and write
   a one-page recommendation. *This is the actual job of an inference
   engineer working with DeepEP.*
4. **Modify DeepEP.** Clone the real repo. Make one concrete change (e.g., add
   a new FP8 scale-sharing strategy, or a different channel schedule). Get it
   to build and pass the existing tests. *If you can do this, you've fully
   graduated.*

---

# DeepEP Connection

This lesson *is* the connection — the synthesis. The lab is complete.

---

# The lab, in one paragraph

You started by copying a buffer between host and one GPU (lesson 1), learned
that peer copies go over NVLink (2), built every collective from broadcast to
ring AllReduce by hand (3–7), met AllToAll — the MoE skeleton (8), built a
lock-free ring buffer (9), mastered streams and events for overlap (10–11),
moved into the NVSHMEM symmetric-heap programming model (12–14), added
routing (15) and a full MoE dispatch (16), re-cast it the DeepEP way with
device-side puts and channels (17), learned to read the real source (18),
climbed the optimization ladder (19), and now see the whole system (20). You
can explain every NCCL collective, implement simplified versions, program
NVSHMEM, and — the goal — **read and modify DeepEP source code with
confidence**.

That was the point. Go build something.
