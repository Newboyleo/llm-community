# Build everything from the top.
cmake_minimum_required(VERSION 3.18)
project(llm_communication_lab LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED NVSHMEM_DIR AND CMAKE_LIBRARY_ARCHITECTURE)
    set(_nvshmem_deb_config
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvshmem/${CUDAToolkit_VERSION_MAJOR}/cmake/nvshmem")
    if(EXISTS "${_nvshmem_deb_config}/NVSHMEMConfig.cmake")
        set(NVSHMEM_DIR "${_nvshmem_deb_config}" CACHE PATH "NVSHMEM CMake package directory")
    endif()
    unset(_nvshmem_deb_config)
endif()

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

if(DEFINED NVSHMEM_DIR)
    foreach(_cuda_arch ${CMAKE_CUDA_ARCHITECTURES})
        string(REGEX REPLACE "-(real|virtual)$" "" _cuda_arch_num "${_cuda_arch}")
        if(_cuda_arch_num MATCHES "^[0-9]+$" AND _cuda_arch_num LESS 70)
            message(FATAL_ERROR
                "NVSHMEM lessons require CMAKE_CUDA_ARCHITECTURES >= 70, "
                "but got ${CMAKE_CUDA_ARCHITECTURES}. Reconfigure with e.g. "
                "`cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=70` or `80`. "
                "If this build directory cached an older value such as 52, remove "
                "the cache or pass -DCMAKE_CUDA_ARCHITECTURES explicitly.")
        endif()
    endforeach()
    unset(_cuda_arch_num)
    unset(_cuda_arch)
endif()

# Common helpers compile into a static lib that every lesson links.
add_subdirectory(common)

# Each lesson is an independent target. Lessons are guarded so a missing
# dependency (e.g. NVSHMEM) does not break the rest of the build.
option(BUILD_ALL_LESSONS "Build every lesson" ON)

# --- Default lessons ---------------------------------------------------------
set(DEFAULT_LESSONS
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
    lesson11-events
    lesson12-nvshmem-basics
    lesson13-nvshmem-ring-buffer
    lesson14-producer-consumer
    lesson15-token-routing
    lesson16-moe-dispatch
)

foreach(lesson ${DEFAULT_LESSONS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${lesson}/CMakeLists.txt")
        add_subdirectory(${lesson})
    endif()
endforeach()

# --- Advanced NVSHMEM lessons ------------------------------------------------
# These require NVSHMEM and are kept optional because they are later DeepEP-style
# exercises. Lessons 12-14 are included in the default lesson list above; their
# own CMake files skip targets cleanly when NVSHMEM is unavailable.
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
            lesson17-mini-deepep
            lesson19-optimizing
        )
        foreach(lesson ${NVSHMEM_LESSONS})
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${lesson}/CMakeLists.txt")
                add_subdirectory(${lesson})
            endif()
        endforeach()
    endif()
endif()

# --- Documentation-only lessons ---------------------------------------------
# Lessons 18 and 20 have no buildable code; they are reading/analysis guides.

message(STATUS "llm-communication-lab configured.")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch:        ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  NVSHMEM lessons:  ${BUILD_NVSHMEM_LESSONS}")
