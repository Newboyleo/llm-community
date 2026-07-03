# Glossary

Concise definitions for terms used across the lab. Cross-link from lesson text
rather than re-defining.

## GPU memory & hardware

- **device pointer** — a pointer valid only on the GPU (`cudaMalloc`).
- **host pointer** — a pointer valid only on the CPU (`malloc`/`new`).
- **unified memory (UVA)** — a single virtual address space across CPU + all
  GPUs; the runtime migrates pages. Enable transparently on 64-bit Linux with
  Pascal+. Lets you pass the same pointer to any device.
- **peer access** — GPU A can directly read/write GPU B's memory without
  bouncing through host memory. Requires `cudaDeviceEnablePeerAccess` or
  (cleaner) `cudaDeviceCanAccessPeer` + the unified-address setup.
- **P2P copy** — `cudaMemcpyPeer` / `cudaMemcpyPeerAsync` between two devices.
  Goes over NVLink if available, else PCIe.
- **NVLink** — NVIDIA's high-bandwidth GPU↔GPU / GPU↔CPU interconnect.
  Modern links: 50 GB/s per direction per link; pairs often have several.
- **NVSwitch** — a crossbar switch that gives full all-to-all bandwidth between
  every GPU in a DGX/HGX node.
- **PCIe root complex / host bridge (PHB)** — when two GPUs sit under different
  host bridges, copies transit host memory and are slow. `nvidia-smi topo -m`
  labels this `PHB` or `SYS`.

## Streams & sync

- **stream** — an ordered queue of GPU work. Work in the same stream runs in
  order; work in different streams may overlap.
- **default stream** — stream 0. With the legacy default, it serializes against
  all other streams unless you compile with `--default-stream per-thread`.
- **event** — a marker recorded into a stream. `cudaStreamWaitEvent` makes a
  stream wait on an event — the primary tool for cross-stream dependencies.
- **fence / barrier** — in this lab, a memory-ordering primitive ensuring prior
  writes are visible before later reads. CUDA's `__threadfence_system()` is the
  heavyweight, system-wide version.

## Collectives (the vocabulary)

- **rank** — one participant in a collective, identified by an integer
  `[0, n)`. In NCCL this is `ncclCommAllRank`. In our hand-rolled collectives
  it's just a GPU index.
- **world size** — the number of ranks (`n`).
- **broadcast** — rank 0's data ends up on all ranks.
- **scatter** — rank 0 splits its data and sends a different piece to each rank.
- **gather** — the reverse of scatter: each rank's piece is assembled on one rank.
- **allgather** — gather, but every rank gets the full result.
- **reduce** — element-wise combine (usually sum) across ranks, result on one rank.
- **allreduce** — reduce, but every rank gets the result.
- **reduce-scatter** — reduce, then scatter the result so each rank owns a slice.
- **alltoall** — rank `i` sends a distinct block to rank `j` for every `(i,j)`.

### Communication volume (per rank) for the ring algorithms

| Collective       | Data sent per rank | Data received per rank |
|------------------|--------------------|------------------------|
| Broadcast (ring) | n·S / n = S        | S                      |
| AllGather (ring) | (n−1)·S            | (n−1)·S                |
| ReduceScatter    | (n−1)·(S/n)        | (n−1)·(S/n)            |
| AllReduce = RS + AG | (n−1)·(S/n) + (n−1)·S … see lesson 7 for the exact accounting |

`S` = total payload size, `n` = world size. The ring's claim to fame is that
each rank only ever sends `((n−1)/n)·S` for AllReduce — bandwidth-optimal.

## NVSHMEM

- **symmetric heap** — memory allocated by `nvshmem_malloc` that has the *same
  virtual address* on every PE (processing element). This is the whole trick:
  any PE's pointer is valid on any PE, so a kernel on PE 0 can `nvshmem_put`
  to PE 1's buffer using PE 1's pointer.
- **PE (processing element)** — NVSHMEM's term for a rank / GPU.
- **put / get** — one-sided write / read into a remote PE's symmetric heap.
  Initiated from the GPU kernel itself, not the host.
- **quiet / fence** — `nvshmem_quiet` waits for all puts to complete;
  `nvshmem_fence` orders puts/gets but does not wait. The most common bug in
  NVSHMEM code is forgetting a `quiet` and reading stale data.

## MoE / DeepEP

- **expert** — a feed-forward sub-network. A MoE layer has E experts; each
  token is routed to a few of them.
- **gate / router** — the small linear layer producing per-expert logits that
  decide which experts each token visits.
- **top-k routing** — each token picks its k highest-gated experts.
- **dispatch** — sending each token's hidden state to the GPU(s) hosting its
  chosen experts. This is the AllToAll at the heart of MoE.
- **combine** — the reverse: gather expert outputs back, weighted by gate
  scores, to reassemble the token stream.
- **capacity / capacity factor** — the buffer size each expert reserves
  (tokens × capacity_factor). DeepEP avoids fixed capacity by computing the
  exact per-expert count up front.
- **DeepEP** — DeepSeek's open-source MoE dispatch/combine library. Goals:
  low-latency dispatch (for prefill) and high-throughput dispatch (for batched
  decode), with optional FP8 quantization.
- **channel** — in DeepEP, an independent communication lane (a stream + a
  chunk of work) used to overlap transfers. Many channels hide latency.

## NCCL internals (referenced in lessons)

- **ring** — the data structure used for AllReduce/ReduceScatter/AllGather.
  Each rank sends and receives one chunk per step; n−1 steps total.
- **tree** — used for Broadcast/Reduce, and as a complement to the ring in
  AllReduce to halve the latency term.
- **channel** — in NCCL, a logical communication path (ring or tree) bound to
  a stream. Multiple channels run in parallel to saturate bandwidth.
- **net / collNet** — NCCL's plugin interface for inter-node (network) transport.
