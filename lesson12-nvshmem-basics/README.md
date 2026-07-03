# Lesson 12 — NVSHMEM Basics

> NVSHMEM gives every GPU a piece of a **symmetric heap** whose pointers are
> valid on *every* GPU. A kernel on GPU 0 can `nvshmem_put` straight into GPU
> 1's memory — no host involvement, no `cudaMemcpyPeer`. This is the
> programming model DeepEP is built on.

---

# Overview

## What are we building?

A first NVSHMEM program. Two (or more) GPUs each allocate the **same-sized**
symmetric buffer. Then:

1. A kernel on PE 0 uses `nvshmem_int_put` to write into PE 1's buffer.
2. A kernel on PE 1 uses `nvshmem_int_get` to read from PE 0's buffer.
3. We `nvshmem_quiet()` to force completion, then verify.

```
   PE 0 (GPU 0)                         PE 1 (GPU 1)
   ┌────────────┐   nvshmem_int_put     ┌────────────┐
   │  src[]     │ ─────────────────────▶│  dst[]     │
   │  (symmetric)│   initiated by PE0's │  (symmetric)│
   └────────────┘   kernel, over NVLink └────────────┘
                   ◀─────────────────────
                       nvshmem_int_get
```

## Why does it matter?

Up to lesson 11, every cross-GPU transfer was orchestrated by the **host**:
the CPU called `cudaMemcpyPeerAsync`. NVSHMEM flips this — the **kernel
itself** initiates the transfer. That means:

- **No launch overhead per transfer.** One kernel can issue thousands of P2P
  writes from the device side, each at ~hundred-nanosecond latency instead of
  ~microsecond host-launch latency.
- **Fine-grained, dynamic routing.** A kernel can decide *per-thread* which PE
  to write to — exactly what MoE dispatch needs (each token goes to a
  different expert PE).
- **Symmetric pointers.** `dst` on PE 1 has the *same virtual address* on PE
  0, so PE 0's kernel can write to it using the pointer it already has.

This is the programming model that makes low-latency MoE dispatch possible.

## Where is it used in LLM inference?

- **DeepEP** is an NVSHMEM application. Its dispatch/combine kernels are
  device-side NVSHMEM put/get sequences over symmetric channel buffers.
- **Multinode tensor parallelism** (NVSHMEM spans nodes via IB/RDMA).
- **PGAS-style** sparse data movement anywhere a host-orchestrated collective
  would be too coarse.

---

# Goal

- Initialize NVSHMEM, allocate symmetric memory, launch an NVSHMEM-aware kernel.
- Use `nvshmem_int_put` / `nvshmem_int_get` from inside a kernel.
- Understand `nvshmem_quiet` vs `nvshmem_fence` — the #1 source of NVSHMEM bugs.
- See that the *same pointer* is valid on every PE.

---

# Background

## Symmetric heap

`nvshmem_malloc(size)` returns a pointer `p` such that `p` refers to a buffer
on *every* PE, all at the **same virtual address**. So if PE 0 has `p`, then
`p` on PE 0 is PE 0's buffer, and `p` on PE 1 is PE 1's buffer — same address,
different physical memory. NVSHMEM's runtime + the GPU's address space layout
make this work.

This is why `nvshmem_int_put(dst, src, n, dst_pe)` works: `dst` is the
*remote* PE's symmetric pointer, which happens to be the same address as the
caller's `dst` would be. You don't compute a remote address; you use the
symmetric one you already have.

## The put/quiet contract

```c
// inside a kernel on PE 0:
nvshmem_int_put(dst_on_pe1, src_local, n, /*dst_pe=*/1);
// ... NOT YET visible on PE 1 ...
nvshmem_quiet();   // wait until ALL prior puts from THIS PE have landed
// NOW visible on PE 1
```

`nvshmem_quiet()` blocks the issuing thread until all its prior puts are
*complete* (delivered to the remote PE). `nvshmem_fence()` is weaker: it
orders puts/gets but doesn't wait for completion. The classic bug:

```c
nvshmem_int_put(dst, src, n, 1);
// no quiet
x = *dst_remote_flag;   // BUG: the put may not have landed; flag read may race
```

## Init model

```c
nvshmem_init();                 // bootstrap
int pe = nvshmem_my_pe();
int npes = nvshmem_n_pes();
int* p = (int*)nvshmem_malloc(N * sizeof(int));
// ... launch kernels that use p ...
nvshmem_free(p);
nvshmem_finalize();
```

For single-node, you launch kernels with the ordinary `<<<>>>` syntax on a
stream (NVSHMEM hooks the CUDA runtime). For multi-node you'd use
`nvshmemx_collective_*_launch`; we stay single-node.

---

# Architecture Diagram

```
   Symmetric heap (same VA on every PE):

        PE 0                                PE 1
   VA: 0x7f...100  ─────────────────  VA: 0x7f...100   (same address!)
        │ buf[0..N]                         │ buf[0..N]
        │ (GPU0 HBM)                        │ (GPU1 HBM)

   Kernel on PE 0:
     nvshmem_int_put(buf, local_src, n, 1)   // writes into PE1's buf, same VA
     nvshmem_quiet()                          // wait for landing
```

---

# Source Code Walkthrough

`nvshmem_basics.cu`:

- `init_pe()` — each PE fills its symmetric buffer with `1000*pe + i`, so we
  can tell buffers apart.
- `put_kernel(int* buf, int n, int dst_pe)` — PE 0 writes a marker into
  `dst_pe`'s buffer.
- `get_kernel(int* buf, int n, int src_pe)` — PE 1 reads `src_pe`'s marker.
- `main()` — init, alloc symmetric, run kernels on each PE (one kernel per
  PE, synchronized by `nvshmem_barrier_all`), verify.

Key lines (device-side put):

```c
__global__ void put_kernel(int* buf, int n, int dst_pe) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        int val = 1000000 + i;           // a marker
        nvshmem_int_put(&buf[i], &val, 1, dst_pe);  // one-sided write to dst_pe
    }
}
// host, after kernel:
nvshmem_quiet();   // not needed inside kernel if kernel ends; but harmless
```

(Note: `nvshmem_quiet` from a host thread waits for that thread's *device-side*
puts too, but the clean pattern is to call `cudaDeviceSynchronize` first, then
inspect.)

---

# Build

NVSHMEM is required. Configure with it on:

```bash
cmake -S . -B build \
      -DBUILD_NVSHMEM_LESSONS=ON \
      -DNVSHMEM_DIR=/path/to/nvshmem
cmake --build build -j --target nvshmem_basics
```

NVSHMEM ships its own CMake package; this lesson's `CMakeLists.txt` links
`NVSHMEM::nvshmem` (or the raw library if the package isn't found).

Running NVSHMEM programs needs the bootstrap; for single-node, the default
"shmem" bootstrap works:

```bash
# NVSHMEM needs CUDA_VISIBLE_DEVICES to map PEs to GPUs.
CUDA_VISIBLE_DEVICES=0,1 ./build/lesson12-nvshmem-basics/nvshmem_basics
```

---

# Run

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/lesson12-nvshmem-basics/nvshmem_basics
```

---

# Expected Output

```
==== lesson 12: NVSHMEM basics ====
n_pes = 2
PE 0 buf[0..3] = [0, 1, 2, 3]
PE 1 buf[0..3] = [1000, 1001, 1002, 1003]

[after PE0 puts markers into PE1]
PE 1 buf[0..3] = [1000000, 1000001, 1000002, 1000003]   OK

[after PE1 gets from PE0 into local]
PE 1 local[0..3] = [0, 1, 2, 3]   OK
```

The key observation: PE 0 wrote into PE 1's buffer using **only PE 0's kernel**.
PE 1's host never issued a `cudaMemcpy`. That's the NVSHMEM difference.

---

# Experiment

1. **Drop the quiet.** Remove the `nvshmem_barrier_all` (which implies a
   quiet) and read PE 1's buffer immediately. You may see stale data — the
   put hasn't landed. This is the lesson-9 fence problem, now across GPUs.
2. **Many small puts.** Have one thread put 1024 individual ints (one
   `nvshmem_int_put` per thread). Compare to one `nvshmem_putmem` of 1024 ints.
   The batched version is far faster — each put has overhead. (DeepEP batches
   aggressively.)
3. **Same pointer, different PE.** Print `buf` on PE 0 and PE 1. Same address.
   Now print `buf[0]` read locally on each — different values. Drive home:
   *same VA, different physical memory*.
4. **3+ PEs.** Run with `CUDA_VISIBLE_DEVICES=0,1,2,3`. Have PE 0 fan out
   different markers to PEs 1,2,3 in one kernel. This is the atom of MoE
   dispatch.

---

# Performance Analysis

- **Per-put latency** from the device side is ~100–500 ns on NVLink — far below
  the ~3–5 µs of a host `cudaMemcpyPeerAsync`. This is why DeepEP can afford
  per-token dispatch.
- **But per-put overhead is non-zero.** Issuing 1 M individual `nvshmem_int_put`
  calls is slower than one `nvshmem_putmem` of 4 MB. The win is when the
  *routing is dynamic* (each put goes to a different PE) — you can't batch
  those into one `putmem`, but you can batch *per destination*.
- **`quiet` cost** scales with the number of outstanding puts. Batching puts
  before a quiet (DeepEP's "dispatch a batch then quiet") is essential.

---

# Exercises

1. **Putmem vs int_put.** Time copying 1 MB to a remote PE with one
   `nvshmem_putmem` vs 256K `nvshmem_int_put`s. Plot the crossover.
2. **Get into registers.** Instead of `nvshmem_int_get` into memory, use
   `nvshmem_int_g` (returns the value) to pull a remote value directly into a
   register. Useful for reading a remote flag.
3. **Round-trip latency.** PE 0 puts a flag to PE 1; PE 1 (polling) puts a
   flag back. Measure the round-trip. This is the latency floor of any
   NVSHMEM-based dispatch (lesson 14 builds on it).

---

# DeepEP Connection

```
Lesson 12  NVSHMEM put/get (device-initiated P2P)
   ↓
Lesson 13  NVSHMEM ring buffer (symmetric head/tail)
Lesson 14  producer/consumer with flag signaling
   ↓
DeepEP     every dispatch write is an nvshmem_put into the destination
           expert's symmetric channel buffer; every "tokens ready" signal is
           a flag put + quiet. The host never touches the data path.
```

When you read DeepEP's `intranode_dispatch` kernel, every
`nvshmem_float_put` / `nvshmem_putmem` you see is exactly today's primitive.
The rest of DeepEP is *scheduling* these puts (lessons 13–17) and *quantizing*
them (lesson 19).
