# Hardware this lab assumes

This is a **single multi-GPU node** lab. Multi-node (RDMA/IB) is mentioned in
the NVSHMEM and DeepEP lessons, but every runnable experiment is intra-node.

## Required (to actually run the code)

- **Linux x86_64** with an NVIDIA driver.
- **CUDA 11.8+** (12.x recommended; NVSHMEM lessons want 12.x).
- **At least 2 NVIDIA GPUs** on the same node, ideally 4–8.
- **NVLink** between them (so the bandwidth numbers in the lessons are real).
  PCIe-only setups will run but with ~5–10× lower bandwidth.
- For lessons 12–17: **NVSHMEM 2.x** (`NVSHMEM_DIR` pointed at the install).

## What "good" looks like

| Tier        | GPUs | Link            | Expected peer-copy BW | Notes                       |
|-------------|------|-----------------|-----------------------|-----------------------------|
| Reference   | 8×   | NVLink (NVSwitch) | 300–500 GB/s        | DGX-class                    |
| Typical     | 4×   | NVLink (mesh)   | 100–250 GB/s          | Workstation                  |
| Minimal     | 2×   | PCIe Gen4 x16   | 20–32 GB/s            | No NVLink; still educational |
| Dev-only    | 1×   | —               | n/a                   | Lessons 1 only               |

## Checking your topology

```bash
nvidia-smi topo -m
```

You want `NV#` cells (NVLink) rather than `SYS`/`PHB` (cross-PCIe-host-bridge)
between every pair of GPUs you'll use. Example good output:

```
        GPU0 GPU1 GPU2 GPU3
 GPU0     X  NV12  NV12  NV12
 GPU1  NV12   X  NV12  NV12
 GPU2  NV12  NV12   X  NV12
 GPU3  NV12  NV12  NV12   X
```

`NV12` = 12 NVLinks between the pair.

## No GPU available?

The lesson `README.md` files are written to be readable end-to-end without
running anything. The build will fail cleanly without CUDA. You can still learn
the entire mental model — the code is short enough to trace by hand.
