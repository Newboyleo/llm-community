# common/ — shared helpers across lessons
#
# Provides:
#   lab::checks    — CUDA error checking macros + a small FAIL() helper
#   lab::timing    — CUDA event-based GPU timer + CPU wall-clock helper
#   lab::print     — pretty array printing for verifying collectives
add_library(lab_checks STATIC checks.cpp)
target_include_directories(lab_checks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab_checks
    PUBLIC
        CUDA::cudart
)
add_library(lab_timing STATIC timing.cpp)
target_include_directories(lab_timing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab_timing PUBLIC lab_checks)
target_link_libraries(lab_timing
    PUBLIC
        lab_checks
        CUDA::cudart
)

add_library(lab_print STATIC print.cpp)
target_include_directories(lab_print PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab_print
    PUBLIC
        CUDA::cudart
)

# Convenience aggregate target.
add_library(lab_common INTERFACE)
target_link_libraries(lab_common INTERFACE lab_checks lab_timing lab_print)
