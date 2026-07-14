# Lesson 12 — NVSHMEM basics.
#
# NVSHMEM is optional and not always installed. This CMake file tries the
# shipped CMake config; if absent, it falls back to a manual link against
# NVSHMEM_DIR. If neither works, the target is not created and the top-level
# build proceeds without this lesson.

find_package(NVSHMEM QUIET HINTS "${NVSHMEM_DIR}" ENV NVSHMEM_DIR)

set(_nvshmem_libs "")
set(_nvshmem_found FALSE)
if(TARGET NVSHMEM::nvshmem)
    set(_nvshmem_libs NVSHMEM::nvshmem)
    set(_nvshmem_found TRUE)
elseif(TARGET nvshmem::nvshmem_host)
    set(_nvshmem_libs nvshmem::nvshmem_host)
    if(TARGET nvshmem::nvshmem_device)
        list(APPEND _nvshmem_libs nvshmem::nvshmem_device)
    endif()
    set(_nvshmem_found TRUE)
elseif(DEFINED NVSHMEM_DIR)
    # Manual fallback.
    set(NVSHMEM_INCLUDE_DIR "${NVSHMEM_DIR}/include")
    find_library(NVSHMEM_LIB nvshmem PATHS "${NVSHMEM_DIR}/lib")
    if(NVSHMEM_LIB)
        set(_nvshmem_libs ${NVSHMEM_LIB})
        set(_nvshmem_found TRUE)
    endif()
endif()

if(_nvshmem_found)
    add_executable(nvshmem_basics nvshmem_basics.cu)
    target_link_libraries(nvshmem_basics PRIVATE lab_common ${_nvshmem_libs})
    if(DEFINED NVSHMEM_INCLUDE_DIR)
        target_include_directories(nvshmem_basics PRIVATE ${NVSHMEM_INCLUDE_DIR})
    endif()
    set_target_properties(nvshmem_basics PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson12-nvshmem-basics)
    message(STATUS "  lesson12-nvshmem-basics: enabled")
else()
    message(STATUS "  lesson12-nvshmem-basics: SKIPPED (NVSHMEM not found)")
endif()
