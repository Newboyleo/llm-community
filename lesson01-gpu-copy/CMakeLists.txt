add_executable(gpu_copy gpu_copy.cu)
target_link_libraries(gpu_copy PRIVATE lab_common)
# The executable name matches the lesson dir name (minus the NN- prefix) so
# `cmake --build build --target gpu_copy` works as the README documents.
set_target_properties(gpu_copy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson01-gpu-copy)
