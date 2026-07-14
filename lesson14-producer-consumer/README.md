# Lesson 14 — Producer / Consumer (Signaled)

> Lesson 13's ring buffer moves data, but the consumer must *poll* the head to
> know data arrived. Real dispatch needs an explicit **"this batch is ready"
> signal** so the consumer can start work the instant a batch lands — and so
> the producer can know when the consumer has finished (for backpressure).
> This lesson adds that signal.

---

# Overview

## What are we building?

A cross-GPU producer/consumer with **explicit ready/done flags**:

- Producer pushes a *batch* of tokens, then `nvshmem_put`s a `ready` flag (with
  a `quiet`).
- Consumer polls the `ready` flag; when set, it processes the batch, then
  `nvshmem_put`s a `done` flag back.
- Producer polls `done` before reusing the buffer slot (the backpressure
  handshake).

```
   PE 0 (producer)                       PE 1 (consumer)
   write batch ──put──▶ slots            poll ready flag
   quiet                                   ▼ ready!
   put ready flag ─────────────────────▶  process batch
   poll done flag                          put done flag ───────▶
   ▼ done!                                (slot can be reused)
   write next batch ──put──▶ slots
```

## Why does it matter?

This is the **dispatch/combine handshake** in miniature. DeepEP's expert GPU
doesn't poll a byte counter — it polls a per-batch "ready" flag, and the
dispatching GPU polls a "done" flag to know when it can reuse the channel
buffer. Without this handshake you either (a) poll data (wasteful, racy) or
(b) barrier globally (kills latency). The flag handshake is the
**low-latency** answer.

## Where is it used in LLM inference?

- **DeepEP dispatch:** per-channel `tokens_ready` flag → expert kernel starts;
  expert kernel ends → `combine_ready` flag → combine runs.
- **Pipeline parallelism:** "stage N's activations are ready" flag → stage
  N+1 begins.
- **Any overlapped producer/consumer** where a global barrier would be too
  coarse.

---

# Goal

- Add `ready`/`done` flag signaling to lesson 13's ring.
- See that one flag per **batch** (not per token) keeps overhead low.
- Build the full request/ack handshake and confirm no data races across
  thousands of batches.

---

# Background

## Why flags, not counter polling?

You *could* have the consumer poll the head counter (lesson 13). But:

1. A counter tells you "N items arrived," not "this specific batch is
   complete and consumable." For variable-size batches you need a delimiter.
2. A flag is a single 4-byte put — cheaper to poll than a counter that may
   race with the producer's writes.
3. Flags generalize to **multiple channels**: each channel has its own
   ready/done pair, so the consumer can service whichever channel fires first.

## The handshake (one batch)

```
Producer:                        Consumer:
  put batch data                   loop: if (*ready_flag == my_seq) break;
  quiet                            (process batch)
  *ready_flag_local = my_seq       *done_flag_local = my_seq
  put ready_flag -> consumer       put done_flag -> producer
                                   quiet
  loop: if (*done_flag == my_seq) break;
  (reuse buffer for next batch)
```

The `seq` number disambiguates batches: the consumer waits for *its* expected
seq, and the producer waits for the matching ack. This avoids ABA problems
when the flag wraps.

---

# Architecture Diagram

```
        PE 0 (producer)                          PE 1 (consumer)
   ┌─────────────────────┐                ┌─────────────────────┐
   │ batch data  ──put──▶│ slots          │ poll ready[seq]     │
   │ quiet                │                │   == seq? process   │
   │ ready[seq] ──put──▶  │ ready flag     │ put done[seq] ──▶   │
   │ poll done[seq]       │                │                     │
   │   == seq? reuse      │ done flag ◀────│                     │
   └─────────────────────┘                └─────────────────────┘
```

---

# Source Code Walkthrough

`producer_consumer.cu`:

- `struct Channel { int* slots; int* ready; int* done; int cap; }` — symmetric.
- `producer_kernel(ch, n_batches, batch_size, cons_pe)` — for each batch: fill
  local buffer, `nvshmem_putmem` to consumer slots, `quiet`, set `ready=seq`,
  `put ready`, poll `done==seq`, advance.
- `consumer_kernel(ch, n_batches, batch_size, out, prod_pe)` — for each batch:
  poll `ready==seq`, read slots, process (here: copy to `out`), set `done=seq`,
  `put done`, advance.

Key shape (producer, per batch):

```c
nvshmem_putmem(ch.slots, local_batch, batch_bytes, cons_pe);
nvshmem_quiet();
int seq = b + 1;                       // 1-based so 0 means "empty"
nvshmem_int_put(ch.ready, &seq, 1, cons_pe);
nvshmem_quiet();
int d;
do { d = nvshmem_int_g(ch.done, cons_pe); } while (d != seq);
```

---

# Build

```bash
cmake -S . -B build -DBUILD_NVSHMEM_LESSONS=ON -DNVSHMEM_DIR=/path/to/nvshmem
cmake --build build -j --target producer_consumer
```

---

# Run

```bash
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/nvshmem_12/nvshmrun -np 2 \
    ./build/lesson14-producer-consumer/producer_consumer
# batches, batch_size
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/nvshmem_12/nvshmrun -np 2 \
    ./build/lesson14-producer-consumer/producer_consumer 100 256
```

---

# Expected Output

```
==== lesson 14: producer/consumer (signaled) ====
batches=100, batch_size=256, capacity=8

PE1 consumed 100 batches
batch 0 first 4: [0, 1, 2, 3]
batch 99 first 4: [99200, 99201, 99202, 99203]
all batches correct: YES
elapsed 4.1 ms
```

---

# Experiment

1. **Drop the done handshake.** Have the producer reuse the slot immediately
   after `ready`, without waiting for `done`. At capacity=1 this corrupts
   immediately (producer overwrites data the consumer hasn't read). At high
   capacity it may survive — find the breaking point.
2. **Variable batch sizes.** Make `batch_size[b]` vary; include the size in
   the ready message (e.g., `ready = seq | (size << 16)`). The consumer must
   parse it. This is the MoE-dispatch shape (variable token counts).
3. **Two channels, one consumer.** PE0 sends to PE1 on channel A and to PE2 on
   channel B. The consumer side (PE1 and PE2) each poll their own flag — fully
   independent. This is the multi-expert pattern.
4. **Wake-up latency.** Time from `ready` put landing to consumer noticing it
   (the poll spin). With a backoff (e.g., `__nanosleep`) you can trade latency
   for power; DeepEP spins flat-out for minimum latency.

---

# Performance Analysis

- **Per-batch cost** = one `putmem` (data) + two `int_put` (ready/done) + two
  `quiet` + two poll spins. The `quiet`s dominate at small batch size.
- **Batching pays double:** larger batches both (a) amortize the quiet over
  more bytes and (b) let the consumer do more work per wake-up, hiding the
  poll-to-start latency.
- **The done handshake costs a round-trip per batch.** At very low latency
  this is the floor. DeepEP hides it by running **many channels in parallel**:
  while channel A waits for `done`, channels B,C,D are streaming. Lesson 17
  assembles this.

---

# Exercises

1. **Merge ready+data.** Put the ready flag *into* the last word of the batch
   (a sentinel). One fewer put per batch. DeepEP does variants of this.
2. **Multiple in-flight batches.** Let the producer have B batches in flight
   (B ≤ capacity) before requiring any `done`. This is *credit-based*
   flow control — the producer doesn't stall until all credits are used.
3. **Measure wake-up.** Instrument the gap between `ready` put and consumer
   loop noticing. It's the latency floor for dispatch-to-expert-compute start.

---

# DeepEP Connection

```
Lesson 13  NVSHMEM ring (data + head/tail)
Lesson 14  + ready/done flag handshake                     <- you are here
   ↓
DeepEP     per-channel:
             dispatch kernel puts tokens -> quiet -> sets "tokens ready"
             expert kernel polls "tokens ready" -> runs -> sets "combine ready"
             combine kernel polls "combine ready" -> runs -> sets "done"
           Many channels in flight smooth the per-channel round-trip.
```

When reading DeepEP in lesson 18, the `notify`/`wait` calls you see on
per-channel flags are exactly today's `ready`/`done` puts and polls. The
"low-latency" in "low-latency dispatch" is the sum of: device-initiated puts
(lesson 12) + ring-buffered streaming (lesson 13) + flag handshake (today) +
many parallel channels (lesson 17) + FP8 quantization (lesson 19).
