# build & run helpers — thin wrappers, intentionally simple.

# Configure the CUDA-only lessons.
configure() {
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "$@"
}

# Build everything (or one lesson: build lesson05-reduce-scatter).
build() {
    cmake --build build -j "$@"
}

# Build & run a single lesson by directory name.
run() {
    cmake --build build -j --target "$1" 2>/dev/null
    "./build/$1/$1" "${@:2}"
}

# Configure with NVSHMEM lessons enabled.
#   NVSHMEM_DIR=/opt/nvshmem configure_nvshmem
configure_nvshmem() {
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_NVSHMEM_LESSONS=ON \
          -DNVSHMEM_DIR="${NVSHMEM_DIR:-/opt/nvshmem}" "$@"
}
