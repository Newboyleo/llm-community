add_executable(streams streams.cu)
target_link_libraries(streams PRIVATE lab_common)
# Per-thread default stream so created streams actually overlap with the
# "default" stream on each thread — the same assumption NCCL/NVSHMEM make.
target_compile_options(streams PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-default-stream;per-thread>)
set_target_properties(streams PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson10-streams)
