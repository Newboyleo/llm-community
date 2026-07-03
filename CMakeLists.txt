# Build everything from the top.
cmake_minimum_required(VERSION 3.18)
project(llm_communication_lab LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Release by default — we benchmark.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Ask for architecture of the building host. Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Common helpers compile into a static lib that every lesson links.
add_subdirectory(common)

# Each lesson is an independent target. Lessons are guarded so a missing
# dependency (e.g. NVSHMEM) does not break the rest of the build.
option(BUILD_ALL_LESSONS "Build every lesson" ON)

# --- CUDA-only lessons (1-11) ------------------------------------------------
set(CUDA_LESSONS
    lesson01-gpu-copy
    lesson02-peer-copy
    lesson03-broadcast
    lesson04-allgather
    lesson05-reduce
    lesson06-reduce-scatter
    lesson07-ring-allreduce
    lesson08-alltoall
    lesson09-ring-buffer
    lesson10-streams
)

foreach(lesson ${CUDA_LESSONS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${lesson}/CMakeLists.txt")
        add_subdirectory(${lesson})
    endif()
endforeach()

# --- NVSHMEM lessons (12-17) -------------------------------------------------
# These require NVSHMEM. Gate behind a find to avoid breaking the CUDA-only build.
option(BUILD_NVSHMEM_LESSONS "Build NVSHMEM lessons" OFF)
if(BUILD_NVSHMEM_LESSONS)
    find_package(NVSHMEM)
    if(NOT NVSHMEM_FOUND)
        message(WARNING
            "BUILD_NVSHMEM_LESSONS=ON but NVSHMEM not found. "
            "Set -DNVSHMEM_DIR=/path/to/nvshmem or install it. "
            "Skipping NVSHMEM lessons.")
    else()
        set(NVSHMEM_LESSONS
            lesson12-nvshmem-basics
            lesson13-nvshmem-ring-buffer
            lesson14-producer-consumer
            lesson15-token-routing
            lesson16-moe-dispatch
            lesson17-mini-deepep
        )
        foreach(lesson ${NVSHMEM_LESSONS})
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${lesson}/CMakeLists.txt")
                add_subdirectory(${lesson})
            endif()
        endforeach()
    endif()
endif()

# --- Documentation-only lessons (18-20) --------------------------------------
# These have no buildable code; they are reading/analysis guides.

message(STATUS "llm-communication-lab configured.")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch:        ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  NVSHMEM lessons:  ${BUILD_NVSHMEM_LESSONS}")
