# Lesson 18 — Reading DeepEP Source

> A guided tour of the real [DeepEP](https://github.com/deepseek-ai/DeepEP)
> source tree. By now every file should feel familiar — you've built teaching
> versions of its pieces in lessons 1–17. This lesson maps our primitives onto
> their production names so you can open the repo and navigate.

---

# Overview

## What are we doing?

No code to build here. We walk the DeepEP repo file-by-file, linking each to
the lesson where you built the teaching version. The goal: when you open
`csrc/kernels/internode_dispatch.cu` (or the intranode analog), you should be
able to say *"ah, that's lesson 17's dispatch kernel, plus FP8, plus N
channels, plus a tuned chunk schedule."*

```
   lessons 1-17  ──build up to──▶  DeepEP source tree
   (this lesson)                    csrc/
                                    csrc/kernels/
                                    csrc/...
```

## Why does it matter?

DeepEP is dense, template-heavy, performance-tuned code. Reading it cold is
hard. But it's *not* doing anything conceptually new beyond lessons 1–17 —
it's doing those things **faster**, with **more knobs**, and **fused**. This
lesson is the bridge: once you see the mapping, the source becomes a reference
implementation of what you already understand.

## Where does it fit in LLM inference?

DeepEP is the dispatch/combine library in DeepSeek-V3-class MoE inference and
training. If you're optimizing a MoE model, this is the code whose knobs
(`num_channels`, `num_max_nvl_chunk`, `fp8`, low-latency vs normal mode) you
tune. Reading it is how you stop treating it as a black box.

---

# Goal

- Navigate the DeepEP repo: know which directory does what.
- Map each major source file to a lesson in this lab.
- Read a representative dispatch kernel and explain each section in terms of
  lessons 12–17.
- Know where to look when tuning a knob or debugging a failure.

---

# Background: DeepEP at a glance

DeepEP provides four user-facing entry points, all in Python wrappers around
CUDA/NVSHMEM kernels:

| Entry point                      | Purpose                                  | Lesson analog |
|----------------------------------|------------------------------------------|---------------|
| `intranode_dispatch` (NVLink)    | dispatch tokens within a node            | 17            |
| `intranode_combine`              | reverse: gather expert outputs           | 16 (reverse)  |
| `internode_dispatch` (RDMA/IB)   | dispatch across nodes                    | 17 + RDMA     |
| `internode_combine`              | reverse across nodes                     | 16 (reverse)  |

Each has a **normal** (throughput) and **low-latency** (small-batch) variant.
The low-latency variant uses fewer, smaller channels and tighter polling;
the normal variant uses many channels and larger chunks.

The repo (approximate layout — verify against the upstream `main` when you
clone it, since names shift):

```
DeepEP/
├── README.md
├── csrc/
│   ├── common.h                  # shared types, macros
│   ├── comm.h / comm.cpp         # NVSHMEM bootstrap, PE/comm handle
│   ├── kernels/
│   │   ├── internode_dispatch.cu
│   │   ├── internode_combine.cu
│   │   ├── intranode_dispatch.cu
│   │   ├── intranode_combine.cu
│   │   ├── kernel_helpers.cu
│   │   └── ...
│   └── pybind.cpp                 # Python bindings
├── deep_ep/
│   ├── buffer.py                  # channel buffer abstraction
│   ├── comm.py                    # Python-side comm init
│   └── ...
└── tests/                         # reference correctness tests
```

---

# Architecture Diagram (source-level)

```
   Python (deep_ep/*.py)
        │  pybind
        ▼
   csrc/pybind.cpp  ── exposes: dispatch(), combine(), low_latency_dispatch(), ...
        │
        ▼
   csrc/comm.*         NVSHMEM init, comm handle  ── lesson 12 (init)
        │
        ▼
   csrc/kernels/
     intranode_dispatch.cu   ── lesson 17 (NVSHMEM puts + flags + channels)
     intranode_combine.cu    ── lesson 16-reverse (combine = inverse dispatch)
     internode_dispatch.cu   ── lesson 17 + RDMA transport
     internode_combine.cu    ── combine over RDMA
        │
        ▼
   csrc/kernels/kernel_helpers.cu
     - prefix sums (lesson 15)         - channel scheduling (lesson 17)
     - flag/quiet discipline (lesson 14) - FP8 pack/unpack (lesson 19)
```

---

# File-by-file walkthrough

> File paths below are the *canonical* DeepEP names; if upstream renamed
> something, search for the kernel name (`intranode_dispatch`,
> `dispatch_buffer`, `num_channels`) — the structure is stable.

## `csrc/comm.*` — NVSHMEM bootstrap  →  **lesson 12**

This is where `nvshmem_init`, `nvshmem_malloc` (symmetric heap), and the
per-PE handle live. **Read this first** — it's exactly lesson 12's init, plus
multi-node bootstrap (NVSHMEM's `nvshmemx_*_init` with an IB/RDMA transport).
The `comm` object holds:

- the symmetric **dispatch buffer** (our `recvbuf`),
- the symmetric **combine buffer**,
- per-channel **ready/done flags** (our `ready`),
- the PE count and this PE's id.

Map: `comm.handle` ≈ our `SymmRing`/`Channel` structs, generalized.

## `csrc/kernels/intranode_dispatch.cu`  →  **lesson 17**

The heart of the library. Read it in this order:

1. **Routing count is already computed** (the Python side ran the gate and
   AllReduced the count matrix — our lesson 15/16). The kernel receives
   `num_tokens_received[src][dst]` and the prefix-sum offsets. *That's our
   `count` and `offsets`.*
2. **Channel loop.** The kernel is launched with a grid covering
   `(channel, dst_chunk)`. Each block handles one slice of one dst's tokens
   on one channel — *exactly our `dim3 grid(nch, n)` from lesson 17*.
3. **The put.** Inside, tokens are read from the source buffer and
   `nvshmem_float_put` / `nvshmem_putmem`'d into the destination's symmetric
   dispatch buffer at the prefix-sum offset. *Our `nvshmem_float_put` loop,
   batched.*
4. **The quiet + flag.** After the chunk's puts, `nvshmem_quiet()` then a
   flag put signals "this channel's chunk for this dst is ready." *Our
   double-quiet discipline from lesson 14.*
5. **FP8 branch.** A `#if`/template selects FP8 packing (4 bytes → 1 byte +
   scale). *Lesson 19's optimization.*

Read the **low-latency** variant next: same structure, fewer channels,
smaller chunks, and the consumer-side polling is tighter (no backoff).

## `csrc/kernels/intranode_combine.cu`  →  **lesson 16 reversed**

The combine is the **transpose** of dispatch: each expert GPU puts its
processed outputs back to the originating token GPUs, using the *transposed*
count matrix (`num_tokens_received` transposed). The combine weights
(`gate_score`) are applied — multiply each expert's output by the token's
gate weight before putting, so the receiver just sums. *This is the
weighted-sum Reduce of lesson 5, distributed across the AllToAll of lesson 8,
with NVSHMEM puts of lesson 17.*

## `csrc/kernels/internode_*.cu`  →  **lesson 17 over RDMA**

Structurally identical to intranode, but the transport is RDMA (InfiniBand)
via NVSHMEM's network backend instead of NVLink. The differences you'll see:

- **Chunk sizes are larger** (RDMA has higher per-message overhead than
  NVLink, so batching matters more).
- **The number of channels is tuned for NIC count**, not NVLink pairs.
- **A "sm split" parameter** controls how SMs are divided between dispatch
  and the (overlapped) expert compute — *lesson 10's overlap, made explicit*.

## `csrc/kernels/kernel_helpers.cu`  →  **lessons 14, 15, 19**

Utility kernels:

- **Prefix sums** over the count matrix (column and row) — *lesson 15*.
- **Flag/quiet helpers** — *lesson 14*.
- **FP8 pack/unpack** — *lesson 19*.
- **Channel/chunk scheduling** — deciding which `(channel, chunk)` each block
  handles. This is where the tuning lives.

## `deep_ep/buffer.py`, `deep_ep/comm.py`  →  **Python orchestration**

The user-facing layer. `buffer.py` wraps the symmetric buffers; `comm.py`
handles init. The dispatch call signature (simplified):

```python
buffer.dispatch(
    send_x, send_topk_idx, send_topk_weights,    # tokens + routing
    num_tokens_received,                          # the count matrix (lesson 15)
    ...,
    num_channels=N,                               # lesson 17's nch
    chunk_bytes=...,                              # lesson 19's chunk tuning
)
```

`num_tokens_received` is computed by the caller (the model code runs the gate,
AllReduces the counts). DeepEP *consumes* the routing decision; it doesn't
make it. This separation is important: **lesson 15 (routing) is the model's
job; lessons 16–17 (moving tokens) are DeepEP's job.**

---

# Build

Nothing to build in this lesson. To **clone and read** the real source:

```bash
git clone https://github.com/deepseek-ai/DeepEP
cd DeepEP
# browse csrc/kernels/intranode_dispatch.cu
```

(Don't try to build it here — it needs a multi-GPU + NVSHMEM + RDMA
environment. Reading is the point of this lesson.)

---

# Run

N/A. This is a reading lesson.

---

# Expected Output

Your "output" is the ability to answer, for any file in `csrc/kernels/`:

- *"Which lesson did I build the teaching version of this in?"*
- *"What's the one new thing this file adds beyond that lesson?"*
- *"Which knob would I turn to make this faster / lower-latency / use less
  memory?"*

If you can answer those for `intranode_dispatch.cu`, you've graduated from
this lesson.

---

# Experiment (reading exercises)

1. **Find the quiet.** Grep `nvshmem_quiet` in `intranode_dispatch.cu`. For
   each occurrence, explain *what put it guards* and *what would race without
   it*. (Every one is a lesson-14 fence.)
2. **Find the channel grid.** Find the `<<<grid, block>>>` launch. Confirm
   `grid` is `(num_channels, something)` — our `dim3 grid(nch, n)`.
3. **Find the prefix sum.** Locate where `num_tokens_received` is turned into
   write offsets. Confirm it's the column-prefix of lesson 15.
4. **Find FP8.** Locate the `#if`/template branch for FP8. Compare the put
   size to the BF16/FP16 branch — it should be ~½ (FP8) or ~¼ (with scale
   sharing) of the BF16 size.
5. **Low-latency vs normal.** Diff the two dispatch entry points. List every
   difference and classify each as "fewer channels," "smaller chunks,"
   "tighter polling," or "different SM split."

---

# Performance Analysis (what to look for)

When reading, notice these production concerns that our teaching version
glossed over:

- **SM partitioning.** DeepEP explicitly reserves some SMs for dispatch and
  the rest for expert compute, so they overlap without thrashing. *Lesson 10
  hinted at this; DeepEP enforces it with launch bounds.*
- **Chunk-size autotuning.** The "chunk" a channel handles isn't fixed; it's
  chosen from `num_max_nvl_chunk` / `num_rdma_chunk` to balance latency
  (small chunks, more parallelism) against overhead (too many chunks → too
  many quiets).
- **Async bundle.** Internode puts go through NVSHMEM's network stream, which
  is a separate engine from the NVLink engine. DeepEP issues both kinds of
  puts in the same kernel to use both engines at once.
- **No host in the data path.** Confirm: the host never calls a per-token
  copy. Everything is device-initiated. *This is the lesson-12 promise,
  delivered.*

---

# Exercises

1. **Annotate a kernel.** Pick `intranode_dispatch.cu`. Add comments mapping
   each section to a lesson number (`// lesson 14: quiet-guarded flag put`).
   This is the single best way to cement the mapping.
2. **Find a bug-shaped comment.** DeepEP has comments warning about ordering
  hazards ("must quiet before…"). Each is a lesson-14 fence. Collect them.
3. **Diff low-latency vs normal.** Write a one-paragraph summary of *why* the
  low-latency version is faster at small batch and slower at large batch,
  grounded in the diffs you found.
4. **Trace one token.** Mentally follow one token from `send_x` on PE 0
  through to the expert kernel on PE 1. Name every kernel, every put, every
  flag. If you can do this, you understand DeepEP.

---

# DeepEP Connection

This lesson *is* the DeepEP connection — the bridge from teaching code to
production code. The remaining lessons:

- **Lesson 19** — the optimization playbook: take our lesson-17 kernel and
  apply DeepEP's tricks (FP8, fused gather, channel tuning, SM split,
  overlap) one at a time, measuring each.
- **Lesson 20** — the full architectural writeup, synthesizing everything.

After lesson 18 you can read the source. After 19 you can *modify* it with
intent. After 20 you can *reason about it as a system*.
