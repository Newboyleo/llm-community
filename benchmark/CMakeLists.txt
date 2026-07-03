# benchmark/ — shared bandwidth / latency harness
#
# This is a tiny library every lesson links when it wants to report numbers in
# a consistent format. Lessons 1-11 use it directly; the NVSHMEM lessons have
# their own copies of the pattern (NVSHMEM init differs enough that a shared
# helper would obscure more than it helps).

add_library(lab_benchmark STATIC benchmark.cpp)
target_include_directories(lab_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab_benchmark PUBLIC lab_common)
