# llm-communication-lab

A hands-on lab for understanding **GPU communication**, **NCCL collectives**, **NVSHMEM**, and **MoE dispatch** — ending with the ability to read and modify [DeepEP](https://github.com/deepseek-ai/DeepEP) source code.

Inspired by **CSAPP** (labs that build understanding from the bottom up) and **MIT 6.824** (distributed systems you implement yourself, not treat as a black box).

> **Philosophy:** Every lesson starts from a runnable experiment. We never say *"NCCL implements AllReduce"* — we **build a tiny AllReduce ourselves**, then compare with NCCL. Theory only ever explains an observation.

---

## Who is this for?

A software engineer who can already write CUDA kernels, but wants to understand:

- how GPUs talk to each other (NVLink, PCIe, P2P, unified memory),
- what each NCCL collective actually does on the wire,
- how NVSHMEM enables symmetric multi-GPU programming,
- how Mixture-of-Experts dispatch works,
- how DeepEP achieves low-latency expert dispatch,
- how to read, modify, and optimize the DeepEP source.

---

## Repository layout

```
llm-communication-lab/
├── README.md            ← you are here
├── docs/                ← cross-cutting notes (hardware, NCCL internals, glossary)
├── common/              ← shared helpers (timing, multi-GPU setup, printing)
├── benchmark/           ← bandwidth / latency harness used across lessons
├── scripts/             ← build & run helpers
└── lessonNN-*/          ← 20 lessons, each independently runnable
```

Each lesson directory is self-contained: it has its own `README.md` (the "lab handout") plus a `CMakeLists.txt` and source. You can `cd` into any lesson and build it without the others.

---

## Curriculum (20 lessons)

The path is deliberately incremental — **one new idea per lesson**.

| #  | Lesson                  | New idea introduced                                     |
|----|-------------------------|---------------------------------------------------------|
| 01 | gpu-copy                | GPU memory spaces + a single `cudaMemcpy`               |
| 02 | peer-copy               | `cudaMemcpyPeer`, P2P, NVLink vs PCIe                   |
| 03 | broadcast               | One rank → all ranks, naive vs tree                     |
| 04 | allgather               | Every rank contributes a slice → every rank has all     |
| 05 | reduce                  | Element-wise sum across ranks → one rank                |
| 06 | reduce-scatter          | Reduce *and* scatter the result (ring warm-up)          |
| 07 | ring-allreduce          | The classic Ring AllReduce (bandwidth-optimal)          |
| 08 | alltoall                | Per-pair exchange; the backbone of MoE dispatch         |
| 09 | ring-buffer             | A lock-free SPSC ring on the GPU                        |
| 10 | streams                 | Concurrency, overlap, and the stream programming model  |
| 11 | events                  | Synchronization, timing, and dependency wiring          |
| 12 | nvshmem-basics          | Symmetric heap, `nvshmem_*` puts/gets                   |
| 13 | nvshmem-ring-buffer     | The ring buffer, reborn over NVSHMEM                    |
| 14 | producer-consumer       | Cross-GPU producer/consumer with signaling              |
| 15 | token-routing           | Gate logits → expert assignment → counts                |
| 16 | moe-dispatch            | Mini MoE dispatch (the full send/recv dance)            |
| 17 | mini-deepep             | A stripped-down DeepEP-style low-latency dispatch       |
| 18 | reading-deepep          | A map of the real DeepEP source tree                    |
| 19 | optimizing              | The optimization playbook (channels, flushes, quant)    |
| 20 | deepep-analysis         | Complete DeepEP architecture writeup                    |

---

## How to use a lesson

Every lesson follows the same template:

1. **Overview** — what we build, why it matters, where it shows up in LLM inference.
2. **Goal** — what you should understand afterwards.
3. **Background** — minimal theory (≤ 2 pages).
4. **Architecture Diagram** — ASCII art of the data flow.
5. **Source Code Walkthrough** — every important function explained.
6. **Build** — exact commands.
7. **Run** — exact commands.
8. **Expected Output** — every line explained.
9. **Experiment** — knobs to turn (tensor size, GPU count, chunk size…).
10. **Performance Analysis** — bandwidth, latency, bottlenecks, *why it's slow*.
11. **Exercises** — small modifications.
12. **DeepEP Connection** — where this idea lives in real DeepEP.

---

## Quick start

```bash
# from the repo root
cmake -S . -B build
cmake --build build -j

# run any lesson, e.g. lesson 1
./build/lesson01-gpu-copy/gpu_copy
```

See [docs/HARDWARE.md](docs/HARDWARE.md) for the hardware this expects
(NVIDIA multi-GPU box with NVLink) and [docs/GLOSSARY.md](docs/GLOSSARY.md)
for terminology.

> **No GPU right now?** The code won't run, but the lab handouts (each lesson's
> `README.md`) are written so you can read them end-to-end and learn the mental
> model. The build will fail cleanly on a machine without CUDA — that's expected.

---

## What you'll be able to do at the end

- explain every NCCL collective from first principles,
- implement simplified versions of each,
- reason about GPU communication hardware (NVLink topology, P2P, unified memory),
- program with NVSHMEM,
- implement a simplified DeepEP,
- **read and modify the DeepEP source code with confidence**,
- understand communication optimization in modern LLM inference systems.

---

## License

MIT. Educational use. DeepEP source references are pointers to the upstream
Apache-2.0 / MIT project; this repo does not bundle DeepEP.
