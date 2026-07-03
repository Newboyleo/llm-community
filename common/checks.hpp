#pragma once
// checks.hpp — minimal CUDA error checking used by every lesson.
//
// We deliberately keep this tiny. The pedagogy of this lab is that the reader
// should see the *real* CUDA calls, not a thick wrapper that hides them. So
// the macros below only exist to turn a returned error code into a clear abort
// with file:line — they do not wrap the calls themselves.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace lab {

// Translate a cudaError_t into a human string. Returns "ok" for success.
inline std::string cuda_err_str(cudaError_t e) {
    return std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
}

// Hard fail with a message. Used by the macros below.
[[noreturn]] inline void fail(const char* file, int line, const std::string& msg) {
    std::fprintf(stderr, "\n[LAB FAIL] %s:%d\n            %s\n\n", file, line, msg.c_str());
    std::fflush(stderr);
    std::abort();
}

}  // namespace lab

// Check a CUDA Runtime call. Usage: LAB_CUDA(cudaMalloc(...));
#define LAB_CUDA(call)                                                       \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            ::lab::fail(__FILE__, __LINE__,                                  \
                        "CUDA call failed: " + ::lab::cuda_err_str(_e));     \
        }                                                                    \
    } while (0)

// Check a kernel launch: catches both launch errors and async errors from
// the kernel itself. Use right after a kernel<<<>>>() invocation.
#define LAB_CUDA_SYNC()                                                      \
    do {                                                                     \
        cudaError_t _e = cudaGetLastError();                                 \
        if (_e != cudaSuccess) {                                             \
            ::lab::fail(__FILE__, __LINE__,                                  \
                        "kernel launch failed: " + ::lab::cuda_err_str(_e)); \
        }                                                                    \
        _e = cudaDeviceSynchronize();                                        \
        if (_e != cudaSuccess) {                                             \
            ::lab::fail(__FILE__, __LINE__,                                  \
                        "kernel sync failed: " + ::lab::cuda_err_str(_e));   \
        }                                                                    \
    } while (0)

// A plain assertion that works on the host.
#define LAB_CHECK(cond, msg)                                                 \
    do {                                                                     \
        if (!(cond)) {                                                       \
            ::lab::fail(__FILE__, __LINE__,                                  \
                        std::string("check failed: (") #cond ") " + msg);    \
        }                                                                    \
    } while (0)
