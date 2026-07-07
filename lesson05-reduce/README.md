# Lesson 05 — Reduce

> Every rank has a buffer. We want the element-wise **sum** of all buffers on
> rank 0 (and only rank 0). The first collective with a *combine* operation,
> not just data movement.

---

# Overview

## What are we building?

A **Reduce** (sum) across `n` GPUs. Each rank `r` has buffer `x_r` of length
`L`. After Reduce, rank 0 holds `x_0 + x_1 + … + x_{n-1}`; other ranks are
unchanged.

We implement:

1. **Naive gather-then-sum:** copy every rank's buffer to rank 0, sum on a
   kernel. Simple, but moves `(n-1)·S` bytes into rank 0's ingress.
2. **Tree reduce:** ranks pairwise-sum up a binary tree. Rank 0 ends with the
   total. Each rank sends `≤ log₂n` times, total data moved `O(S)`.
3. **Ring reduce (one-way):** warm-up for ReduceScatter (lesson 6).

```
Naive:          x1,x2,...,x_{n-1} ──▶ rank0 ──▶ kernel sum
Tree:           (x0+x1) + (x2+x3) + ...  combined pairwise up the tree
```

## Why does it matter?

Reduce introduces the **combine** operation into the data path. Up to now
every lesson moved bytes verbatim; now bytes get *folded* along the way. This
folding is what makes AllReduce possible (a Reduce that everyone gets a copy
of), and AllReduce is the heartbeat of data-parallel training and the
post-MoE combine.

## Where is it used in LLM inference?

- **Gradient averaging** in data-parallel training (the classic AllReduce use;
  Reduce-to-zero is the half-step).
- **Loss/accumulator reduction** across pipeline stages.
- DeepEP **combine** is a Reduce-then-route: expert outputs are weighted-summed
  back into the originating token positions.

---

# Goal

- Implement reduce-sum three ways.
- See the tree's `O(S)` data movement beat naive's `O(n·S)` ingress into rank 0.
- Internalize that **the sum kernel runs on the receiving rank** — combining
  happens at the destination, not in transit.

---

# Background

## Where does the sum happen?

A peer copy moves bytes; it does *not* add. So every "reduce" is really:

```
   sender GPU --copy--> receiver GPU --kernel--> receiver adds into its buffer
```

The tree's efficiency comes from *folding early*: rank 0 and rank 1 each add
their buffers locally first (no copy), *then* rank 1 ships its partial sum to
rank 0 which adds again. The shipped payload is `S` once, not `(n-1)·S`.

## Tree reduce (n=4)

```
leaves:   x0   x1   x2   x3
step 0:   x0 ── +x1 ──▶ x0'   x2 ── +x3 ──▶ x2'    (rank1,3 send partials; 0,2 add)
step 1:   x0' ── +x2' ──▶ x0''                       (rank2 sends to rank0; rank0 adds)
result:   x0'' = x0+x1+x2+x3  on rank 0
```

`log₂n` steps, each step every active link carries `S` bytes once. Total moved
`≈ S · log₂n / 2` (only half the ranks send per step).

---

# Architecture Diagram

```
   Tree Reduce (n=4), arrows = "send my buffer; receiver adds into theirs"

   step 0:   r1 ──▶ r0      r3 ──▶ r2
   step 1:   r2 ──▶ r0
   result:   r0 holds sum of all
```

---

# Source Code Walkthrough

`reduce.cu`:

- `__global__ void add_into(int* dst, const int* src, int L)` — `dst[i] += src[i]`.
  This is the *only* combine primitive we need.
- `reduce_naive` — copy every `r>0`'s buffer to a scratch on rank 0, then run
  `add_into` for each.
- `reduce_tree` — for `step = 0..log2(n)-1`: rank `r` with bit `step` set sends
  its buffer to rank `r ^ (1<<step)` (its tree partner), which runs `add_into`.
  After `log2(n)` steps rank 0 holds the total.

Key shape (tree):

```c
for (int step = 0; (1 << step) < n; ++step) {
    for (int r = 0; r < n; ++r) {
        int partner = r ^ (1 << step);
        if (partner >= n) continue;
        if (r > partner) {            // r sends, partner receives+adds
            cudaMemcpyPeerAsync(scratch[partner], partner, d[r], r, bytes, streams[r]);
        }
    }
    // barrier, then receivers run add_into
    for (int r = 0; r < n; ++r)
        if ((r ^ (1<<step)) < n && r < (r ^ (1<<step)))
            add_into<<<...>>>(d[r], scratch[r], L);
    cudaDeviceSynchronize();
}
```

---

# Build

```bash
cmake -S . -B build && cmake --build build -j --target reduce
```

---

# Run

```bash
./build/lesson05-reduce/reduce
./build/lesson05-reduce/reduce 262144
```

---

# Expected Output

```
==== lesson 05: reduce ====
n_gpus = 4
L = 262144 ints (1.00 MiB per rank)
x0[0..3] = [0, 1, 2, 3]   (rank r holds value = r at every index, so sum = 0+1+2+3 = 6)

==== naive gather-then-sum ====
rank0 sum[0..3] = [6, 6, 6, 6]   OK
0.51 ms

==== tree reduce ====
rank0 sum[0..3] = [6, 6, 6, 6]   OK
0.18 ms
```

---

# Experiment

1. **Change the op.** Replace `add_into` with `max_into` (`dst[i] = max(dst[i], src[i])`).
   The schedule is identical; only the combine kernel changes. This is exactly
   how NCCL parameterizes `ncclReduce` with `ncclMax`/`ncclMin`/`ncclSum`.
2. **Vary n.** Tree's step count is `log₂n`; naive's ingress is `n-1`. At n=8
   the gap is large.
3. **Where's the bottleneck?** Profile with `nsys`. For small `L` you'll see
   the kernel launches and barriers dominate, not the copies. For large `L`
   the copies dominate. This shapes NCCL's tree-vs-ring choice.

---

# Performance Analysis

- **Naive** forces rank 0 to ingest `(n-1)·S` bytes through its ingress links.
  On a mesh those links are shared, so effective bandwidth per sender drops as
  n grows.
- **Tree** distributes the reduction across `log₂n` levels; each level moves
  `S` once per active pair. Total ingress into any one rank is `≤ 2S`. The
  price is `log₂n` serialized steps (each with a barrier), so for *small* `S`
  tree can be *slower* than naive due to launch overhead.
- **The combine kernel is free** for small `L` (memory-bound, runs at HBM
  speed, ~2 TB/s) but adds a synchronization point. NCCL fuses the copy and
  the add into a single kernel to hide this — see lesson 19.

---

# Exercises

1. **AllReduce = Reduce + Broadcast.** Run today's Reduce, then lesson 3's
   Broadcast on the result. Time the pair vs a single ring AllReduce (lesson 7).
   The pair is `O(log n + (n-1))`; the ring is `O(2(n-1))`. For small n the
   pair wins; for large n the ring wins.
2. **Fuse copy + add.** Instead of copy-then-`add_into`, write a kernel on the
   receiver that pulls from the sender's pointer (requires UVA + peer access)
   and adds directly. One kernel, one synchronization. This is the NCCL trick.
3. **In-place vs scratch.** Today we use scratch on the receiver. Try
   in-place: receiver adds the incoming copy directly into its own buffer.
   Saves `S` of HBM per rank.

---

# DeepEP Connection

```
Lesson 05  Reduce (sum, tree)
   ↓
NCCL       reduce() — tree, fused copy+add, multi-channel
   ↓
DeepEP     combine kernel: per-token weighted sum of expert outputs.
           The "reduce" here is over *experts visited by a token*, not over
           ranks — but the mechanical pattern (copy partial → add at destination)
           is identical. DeepEP fuses it with the gather routing.
```

The combine's weighted sum (`out[t] = Σ_e g[e] · expert_e(t)`) is, per token,
a tiny tree-reduce over the ≤k experts that token visited. Today's tree reduce
is the k=1, uniform-routing ancestor.

这一段实际上是 **Tree Reduce 如何自动找到通信伙伴（partner）** 的核心技巧。

关键代码只有一行：

```cpp
partner = r ^ (1 << step);
```

这里的 `^` 不是乘方，而是 **按位异或（XOR）**。

---

## 为什么用 XOR？

先看 4 个 GPU：

```text
Rank:
0
1
2
3
```

它们的二进制编号分别是：

| Rank | Binary |
| ---- | ------ |
| 0    | 00     |
| 1    | 01     |
| 2    | 10     |
| 3    | 11     |

Tree Reduce 每一轮只改变 **一个 bit**。

`1<<step` 就表示要翻转哪一位。

---

## 第一轮：step = 0

```cpp
1 << 0 = 1
```

二进制就是

```text
01
```

于是

```cpp
partner = r ^ 01
```

也就是说：

**把最低位翻转。**

例如：

### Rank 0

```text
00

XOR

01

=

01
```

得到

```text
partner = 1
```

---

### Rank 1

```text
01

XOR

01

=

00
```

得到

```text
partner = 0
```

---

### Rank 2

```text
10

XOR

01

=

11
```

得到

```text
partner = 3
```

---

### Rank 3

```text
11

XOR

01

=

10
```

得到

```text
partner = 2
```

所以第一轮自动形成

```text
0 <-> 1

2 <-> 3
```

画出来就是

```text
0 ----- 1

2 ----- 3
```

每一对距离都是 **1**。

---

## 第二轮：step = 1

现在

```cpp
1<<1 = 2
```

二进制

```text
10
```

于是

```cpp
partner = r ^ 10
```

即翻转第二位。

例如

### Rank0

```text
00

XOR

10

=

10
```

得到

```text
2
```

---

### Rank1

```text
01

XOR

10

=

11
```

得到

```text
3
```

---

### Rank2

```text
10

XOR

10

=

00
```

得到

```text
0
```

---

### Rank3

```text
11

XOR

10

=

01
```

得到

```text
1
```

于是得到

```text
0 <-> 2

1 <-> 3
```

画出来就是

```text
0 --------- 2

1 --------- 3
```

距离变成 **2**。

---

## 为什么只让 `r > partner` 发送？

如果不限制：

例如第一轮

```text
0 -> 1

1 -> 0
```

两个人都会发。

结果双方都收到数据。

然后双方都相加。

整个 Reduce 就乱了。

实际上我们希望：

**一方发送，一方接收。**

因此规定

```cpp
if (r > partner)
```

发送。

否则接收。

---

第一轮：

```text
pair

0 <->1
```

因为

```text
1>0
```

所以

```text
1 ---> 0
```

不是

```text
0 --->1
```

第二组

```text
2<->3
```

因为

```text
3>2
```

所以

```text
3 --->2
```

因此第一轮就是

```text
1 ---->0

3 ---->2
```

---

第二轮：

pair

```text
0<->2
```

因为

```text
2>0
```

所以

```text
2 ---->0
```

而不是

```text
0 ---->2
```

最终整个通信过程就是

```text
        step0

1 -------->0

3 -------->2


        step1

2 -------->0
```

最后得到

```text
      1
       \
        \
         0
        /
       /
      2
     /
    /
   3
```

或者更直观地画成归约树：

```text
        0
       / \
      1   2
          |
          3
```

数据流是：

```text
3
│
▼
2
│
▼
0
▲
│
1
```

最终：

* Rank 1 的数据先合并到 Rank 0；
* Rank 3 的数据先合并到 Rank 2；
* Rank 2 再带着 **(2+3)** 的部分和发送给 Rank 0；
* Rank 0 得到 **(0+1)+(2+3)**，完成 Reduce。

---

### 为什么 XOR 是 Tree 通信中最经典的写法？

`partner = rank ^ (1 << step)` 有三个重要优点：

1. **无需预先构建树结构**，每个 Rank 仅凭自己的编号和当前 `step` 就能计算出通信伙伴。
2. **每一轮只翻转一个二进制位**，通信距离依次为 `1、2、4、8...`，天然形成一棵二叉归约树。
3. **算法完全对称、可扩展**，无论是 Reduce、Broadcast、AllReduce，还是许多 MPI/NCCL 的树形算法，都广泛采用这种基于 XOR 的伙伴选择方式。
