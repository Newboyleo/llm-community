#!/usr/bin/env bash
# bootstrap.sh — one-shot configure + build + smoke test for the CUDA lessons.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] configure"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "$@"

echo "[2/3] build"
cmake --build build -j

echo "[3/3] smoke test (lesson01)"
if [[ -x build/lesson01-gpu-copy/gpu_copy ]]; then
    ./build/lesson01-gpu-copy/gpu_copy
else
    echo "lesson01 binary not found (no CUDA on this host?) — build artifacts above."
fi
