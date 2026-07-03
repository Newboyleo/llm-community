# Lesson 02 — Peer Copy (`cudaMemcpyPeer`)

> The host steps out of the data path. GPU 0 writes directly into GPU 1's HBM
> over NVLink. This is the physical primitive underneath every collective in
> the rest of the course.

---

# Overview

## What are we building?

A program that copies a buffer from **GPU 0's HBM to GPU 1's HBM** three ways:

1. **Naive (through host):** GPU0 → host → GPU1. Two PCIe trips. Slow.
2. **Peer access (`cudaMemcpyPeer`):** GPU0 → GPU1 directly over NVLink.
3. **Unified Virtual Addressing (UVA) pointer:** same physical transfer, but
   the source/dest pointers are interchangeable because of UVA.

```
   NAIVE (host bounce)               PEER (NVLink)
   GPU0 ──PCIe──▶ HOST ──PCIe──▶ GPU1     GPU0 ══════NVLink══════▶ GPU1
        2× PCIe latency + 2× BW            1× NVLink latency + 1× BW
```

## Why does it matter?

`cudaMemcpyPeer` (and the underlying P2P / NVLink path) is **the** primitive
that NCCL, NVSHMEM, and DeepEP are built on. Every collective in this course is
just a *schedule* of peer copies. If you understand:

- that peer access must be **enabled** before it's used,
- that the transfer runs on a **stream** and can be **async**,
- that **NVLink and PCIe give wildly different bandwidth**,

…then you understand the physical layer of GPU communication.

## Where is it used in LLM inference?

- **TP all-gather / reduce-scatter** between pipeline-stages or tensor-parallel
  ranks: every step is a peer copy.
- **MoE dispatch/combine** (lessons 16–17): per-token peer copies to the GPU
  hosting the chosen expert.
- **KV-cache migration** during context-window rollover or expert offloading.

---

# Goal

After this lesson you should be able to:

- enable peer access and explain what happens if you forget,
- predict whether a copy goes over NVLink or PCIe from `nvidia-smi topo -m`,
- explain why the naive host-bounce is ~2× slower than peer copy,
- measure peer bandwidth and compare it to lesson 01's D2D-same-device number.

---

# Background

## Enabling peer access

```c
int can = 0;
cudaDeviceCanAccessPeer(devA, devB, &can);   // is P2P even possible?
cudaSetDevice(devA);
cudaDeviceEnablePeerAccess(devB, 0);          // turn it on (directional!)
```

Peer access is **directional** and must be enabled on *both* devices if you
want bidirectional transfer. Forgetting it doesn't error — the driver silently
falls back to the host-bounce path, which is exactly the bug that makes "my
collective is slow" so hard to debug.

## UVA: why peer pointers "just work"

With Unified Virtual Addressing (default on 64-bit Linux since CUDA 4), every
device pointer is unique across the whole system. So `cudaMemcpyPeer` can be
replaced by a plain `cudaMemcpy(d1, d0, bytes, cudaMemcpyDeviceToDevice)` once
peer access is enabled — the runtime figures out the source and destination
devices from the pointers' virtual addresses. We show both forms.

## Bandwidth you should expect

| Link                | One-way BW        | Latency  |
|---------------------|-------------------|----------|
| PCIe Gen4 x16       | ~25 GB/s          | ~5–10 µs |
| PCIe Gen5 x16       | ~50 GB/s          | ~3–5 µs  |
| NVLink (per pair)   | 100–300 GB/s      | ~1–3 µs  |
| NVSwitch (DGX)      | up to ~450 GB/s   | ~1–2 µs  |

---

# Architecture Diagram

```
      ┌───────────────┐                      ┌───────────────┐
      │    GPU 0      │      NVLink          │    GPU 1      │
      │  d_src[]      │ ═══════════════════▶ │  d_dst[]      │
      │  (HBM)        │   cudaMemcpyPeer     │  (HBM)        │
      └───────────────┘   Async, on stream   └───────────────┘
              ▲                                           │
              │            ┌───────────┐                  │
              └──PCIe──────│   HOST    │─────PCIe─────────┘
                (only the  │ (only for │   (naive path: 2 trips)
                 fallback) │  control) │
                            └───────────┘
```

---

# Source Code Walkthrough

`peer_copy.cu`:

- `enable_all_peers(n)` (from `common/multigpu.hpp`) — calls
  `cudaDeviceEnablePeerAccess` for every supported pair.
- `bench_peer(dev_src, dev_dst, bytes, through_host)` — allocates on both
  devices, times the copy, reports bandwidth. `through_host=true` forces the
  naive path by *disabling* peer access before the copy.
- `main()` — prints the peer matrix, then benches naive vs peer vs UVA forms.

Key lines:

```c
// peer (NVLink) path — must be enabled first
cudaSetDevice(0);
cudaDeviceEnablePeerAccess(1, 0);
cudaMemcpyPeer(d_dst_on_1, 1, d_src_on_0, 0, bytes);

// UVA form — identical transfer, runtime infers devices from pointers
cudaMemcpy(d_dst_on_1, d_src_on_0, bytes, cudaMemcpyDeviceToDevice);

// naive path — host bounce. Slow on purpose.
cudaMemcpy(h_stage, d_src_on_0, bytes, cudaMemcpyDeviceToHost);
cudaMemcpy(d_dst_on_1, h_stage, bytes, cudaMemcpyHostToDevice);
```

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target peer_copy
```

---

# Run

```bash
./build/lesson02-peer-copy/peer_copy
# pick a size
./build/lesson02-peer-copy/peer_copy 67108864   # 64 MiB of ints
```

---

# Expected Output

```
==== lesson 02: peer copy ====
[peers] device topology (2 GPUs):
        GPU0   GPU1
GPU0     X     P2P
GPU1    P2P     X
  (run `nvidia-smi topo -m` to see NVLink vs PCIe)

bytes = 64.00 MiB

==== naive: GPU0 -> HOST -> GPU1 ====
copy 1.97 ms  ... 33840.0 GB/s
copy 2.03 ms  ...
=> 2.00 ms  ~32 GB/s   (two PCIe trips, ~16 GB/s each leg effective)

==== peer: GPU0 -> GPU1 (cudaMemcpyPeer) ====
copy 0.21 ms  ... 314000.0 GB/s
=> 0.21 ms  ~305 GB/s  (NVLink)

==== UVA: plain cudaMemcpy DeviceToDevice ====
copy 0.21 ms  (~same as peer — same physical path)
```

(Numbers are illustrative; H100↔H100 over NVLink sees ~300 GB/s one-way for
large transfers.)

---

# Experiment

1. **Disable peer access, rerun.** Comment out `enable_all_peers`. The peer
   number should drop to ~the naive number — the runtime fell back to the
   host-bounce. *This is the #1 silent perf bug in CUDA communication code.*
2. **Vary size.** At what size does NVLink saturate? (Usually ~1–4 MiB.) Below
   that, you're latency-bound — relevant for MoE dispatch where tokens are
   small.
3. **Read the topology.** Run `nvidia-smi topo -m`. If you see `SYS` or `PHB`
   between your two GPUs instead of `NV#`, peer access is unavailable and
   you'll never beat PCIe. On a real training/inference node you want all-`NV`.
4. **Both directions at once.** Open two terminals, run peer copy 0→1 in one
   and 1→0 in the other simultaneously. NVLink is full-duplex; you should see
   ~2× aggregate. (We exploit this in the ring, lesson 7.)

---

# Performance Analysis

The naive path is ~2× a single PCIe trip because it *is* two PCIe trips,
serialized. The peer path removes both the second trip and the host staging,
so it's bounded only by NVLink.

The gap between this lesson's ~300 GB/s and lesson 01's same-device D2D of
~1.5–2 TB/s is the **HBM↔NVLink asymmetry**: HBM is ~5–8× faster than even
NVLink. That asymmetry is *why* collectives are hard — moving data between GPUs
is always the bottleneck, never the compute on a single GPU.

> **Latency note:** for 4 KiB transfers, peer-copy latency is dominated by the
> ~1–3 µs API + DMA setup, not bandwidth. DeepEP's low-latency dispatch path
> exists precisely to drive this small-message latency down (lesson 17, 19).

---

# Exercises

1. **Async + overlap.** Use `cudaMemcpyPeerAsync` on two streams to run 0→1 and
   1→0 concurrently. Measure aggregate bandwidth — it should approach 2×
   one-way. (This is the kernel of the ring AllReduce, lesson 7.)
2. **3+ GPUs.** Extend to 4 GPUs and copy 0→1, 1→2, 2→3 in a chain on one
   stream. The total time should be ~3× a single hop *if* they serialize — but
   on NVSwitch they may overlap. Find out.
3. **Verify correctness.** After the peer copy, launch a kernel on GPU 1 that
   reads `d_dst` and writes a marker. Confirm via D2H that the data arrived
   intact. (Sounds trivial; it's the foundation of every collective's
   correctness check.)

---

# DeepEP Connection

```
Lesson 02  cudaMemcpyPeer 0→1                  one buffer, one hop
Lesson 08  AllToAll                            N×N scheduled peer copies
Lesson 17  mini-DeepEP                         peer copies + ring buffers + signals
   ↓
DeepEP     intranode_dispatch                  NVLink P2P writes (exactly this primitive)
DeepEP     internode_dispatch                  same idea, over RDMA instead of NVLink
```

DeepEP's `intranode_dispatch` is, at the physical layer, a *schedule of
`cudaMemcpyPeer`-equivalent NVLink writes* — one per (src GPU, dst GPU, token
chunk) tuple. The "cleverness" of DeepEP is entirely in:

- **which** writes to issue (the routing — lessons 15–16),
- **how many in flight at once** (channels + ring buffers — lessons 9, 13),
- **how to signal completion without a host round-trip** (NVSHMEM quiet/fence
  + flags — lessons 12–14).

Today's lesson is the atom; everything else is molecules.
