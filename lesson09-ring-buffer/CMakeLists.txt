add_executable(ring_buffer ring_buffer.cu)
target_link_libraries(ring_buffer PRIVATE lab_common)
set_target_properties(ring_buffer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson09-ring-buffer)
