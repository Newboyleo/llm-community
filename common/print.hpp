#pragma once
// print.hpp — tiny pretty-printers for verifying collectives.
//
// When a lesson says "GPU0 should end with [1,2,3,4]", we want a one-liner that
// prints exactly that. These helpers print the first N elements of a host or
// device buffer with a rank/label prefix.

#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace lab {

// Print up to `n` elements of a host buffer with a label.
template <typename T>
void print_host(const char* label, const T* h, size_t n, size_t max_show = 16) {
    std::printf("%-12s [", label);
    size_t show = std::min(n, max_show);
    for (size_t i = 0; i < show; ++i) {
        std::printf("%g", static_cast<double>(h[i]));
        if (i + 1 < show) std::printf(", ");
    }
    if (n > show) std::printf(", ... (%zu total)", n);
    std::printf("]\n");
}

// Copy a device buffer to a host vector and print it.
template <typename T>
std::vector<T> to_host(const T* d, size_t n) {
    std::vector<T> h(n);
    if (n) cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

// Print a device buffer by copying to host first.
template <typename T>
void print_device(const char* label, const T* d, size_t n, size_t max_show = 16) {
    std::vector<T> h = to_host(d, n);
    print_host(label, h.data(), h.size(), max_show);
}

// A small banner used between sections of lesson output.
inline void banner(const char* msg) {
    std::printf("\n==== %s ====\n", msg);
}

}  // namespace lab
