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

Running NVSHMEM programs needs the NVSHMEM launcher so multiple PEs are
created. This lesson needs at least two PEs:

```bash
# NVSHMEM needs CUDA_VISIBLE_DEVICES to map PEs to GPUs.
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/nvshmem_12/nvshmrun -np 2 \
    ./build/lesson12-nvshmem-basics/nvshmem_basics
```

If the machine has InfiniBand devices but no `nvidia_peermem`/`nv_peer_mem`
kernel module, NVSHMEM may print an IB DMA-BUF probe warning before falling back
to the local GPU path. For this single-node lesson, disable that probe:

```bash
CUDA_VISIBLE_DEVICES=0,1 NVSHMEM_IB_DISABLE_DMABUF=1 \
    /usr/bin/nvshmem_12/nvshmrun -np 2 \
    ./build/lesson12-nvshmem-basics/nvshmem_basics
```

---

# Run

```bash
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/nvshmem_12/nvshmrun -np 2 \
    ./build/lesson12-nvshmem-basics/nvshmem_basics
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


这是一个非常好的问题。**很多人以为 NVSHMEM 比 P2P 快是因为带宽更高，其实不是。**

对于两张通过 NVLink 连接的 GPU 来说：

* **P2P (`cudaMemcpyPeerAsync`)**
* **NVSHMEM (`nvshmem_put/get`)**

底层走的都是 **NVLink（或者 PCIe/RDMA）**。

因此：

> **纯数据传输带宽和链路延迟几乎是一样的。**

NVSHMEM 真正减少的不是**传输时间**，而是**控制和同步开销**。

---

## 先看传统 P2P

假设 GPU0 算完数据，要发送给 GPU1。

流程通常是：

```text
GPU0 kernel
      │
      ▼
CPU 等 kernel 结束
      │
      ▼
cudaMemcpyPeerAsync()
      │
      ▼
DMA Engine 开始搬运
      │
      ▼
CPU/Event 通知 GPU1
      │
      ▼
GPU1 kernel 开始计算
```

这里至少有几个额外步骤：

```
GPU → CPU
CPU 发起 memcpy
CPU 设置 DMA
CPU 发 Event
GPU1 等 Event
```

虽然每一步只有几微秒，但是如果：

```
每个 token
每层
每个专家
```

都来一次，就会累积。

例如：

```
100 层
×
8 GPU
×
每层一次 memcpy

≈ 几百次 Host API
```

CPU 就变成瓶颈。

---

## NVSHMEM

NVSHMEM 可以写成

```cpp
__global__ void kernel(...) {

    compute();

    nvshmem_put(...);

    compute_next();
}
```

整个流程：

```text
GPU0 kernel
      │
      ▼
compute
      │
      ▼
nvshmem_put()
      │
      ▼
NVLink
      │
      ▼
GPU1 内存
```

CPU 完全没有参与。

少掉了：

```
kernel launch
↓

cudaMemcpyPeerAsync

↓

CPU 调度

↓

Event

↓

再次 launch kernel
```

---

## 一个简单的时间线

### P2P

```
GPU0

compute
========

CPU

        launch memcpy
-----------------------

DMA

              =========

GPU1

                       wait
-----------------------
                       compute
                       =======
```

GPU1 必须等 CPU。

---

### NVSHMEM

```
GPU0 kernel

compute
========

put
----

继续 compute
===========
```

GPU0 自己就在 kernel 内完成通信。

GPU1 甚至可以：

```cpp
while(flag==0)
    ;

继续计算
```

或者：

```cpp
nvshmem_wait_until(...)
```

等待数据。

整个控制流都在 GPU。

---

# Kernel Launch 数量减少

例如一个流水线。

P2P：

```
Kernel1

↓

cudaMemcpyPeer

↓

Kernel2

↓

cudaMemcpyPeer

↓

Kernel3
```

要 Launch：

```
Kernel
Memcpy
Kernel
Memcpy
Kernel
```

而 NVSHMEM：

```
Kernel

compute

↓

put

↓

compute

↓

put

↓

compute
```

整个就是

```
一个 Kernel
```

Launch overhead 可以减少很多。

Kernel Launch 一般：

```
3~10 us
```

如果：

```
1000 次通信
```

就已经：

```
3~10 ms
```

---

# GPU 不需要 Host Wakeup

传统 CUDA：

```
GPU

完成

↓

CPU 收到 Event

↓

CPU Launch 下一步
```

NVSHMEM：

```
GPU

完成

↓

GPU 自己继续
```

省去了：

```
GPU→CPU

CPU→GPU
```

来回通知。

---

# 对 MoE 的影响最大

例如 DeepEP dispatch。

传统：

```
Expert0

↓

cudaMemcpyPeer

↓

Expert1

↓

cudaMemcpyPeer

↓

Expert2
```

GPU 一直停下来等 CPU。

NVSHMEM：

```
Expert0

↓

put()

↓

继续处理下一批 token
```

GPU 不停。

---

# KV Cache Transfer

例如 Decode GPU：

传统：

```
CPU

↓

cudaMemcpyPeerAsync

↓

GPU 收到 KV

↓

Decode
```

NVSHMEM：

```
Prefill GPU

↓

put(KV)

↓

Decode GPU

wait_until()

↓

继续 Decode
```

Host 完全退出。

Mooncake、DeepEP 基本都是这种模式。

---

# 真正节省的是哪些时间？

如果把一次通信拆开：

```
总耗时

=

启动开销
+
同步开销
+
真正的数据传输
```

其中：

```
启动

Kernel Launch
API 调用
CPU 调度
```

以及

```
同步

cudaEvent
cudaStreamWaitEvent
Host Synchronize
```

NVSHMEM 基本都可以省掉。

但是：

```
NVLink 上传输 4 MB

需要多少时间？
```

这个：

```
P2P

≈ NVSHMEM
```

因为底层还是同一条 NVLink。

---

## 总结

NVSHMEM 并不是让 **NVLink 更快**，而是让 **GPU 能自己发起通信和同步**。相比传统 P2P，它主要减少的是：

| 开销                             | `cudaMemcpyPeerAsync` | NVSHMEM          |
| ------------------------------ | --------------------- | ---------------- |
| CPU 调用 `cudaMemcpyPeerAsync`   | ❌ 需要                  | ✅ 不需要            |
| CPU 发起 DMA 传输                  | ❌ 需要                  | ✅ GPU 内核直接发起     |
| 多次 Kernel Launch               | ❌ 通常需要                | ✅ 可合并到同一个 Kernel |
| Host 事件调度 (`cudaEvent`、CPU 唤醒) | ❌ 常见                  | ✅ 大幅减少           |
| 数据传输时间（NVLink/PCIe）            | ≈ 相同                  | ≈ 相同             |

因此，**对于一次大块数据（例如 1 GB）的复制，NVSHMEM 不一定比 P2P 更快**；真正的优势出现在**大量小消息、频繁通信、细粒度流水线和 GPU 间协同计算**（如 MoE、KV Cache、Pipeline Parallel）中，因为它显著降低了 CPU 参与和同步带来的累计开销。
