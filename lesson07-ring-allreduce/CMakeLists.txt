add_executable(ring_allreduce ring_allreduce.cu)
target_link_libraries(ring_allreduce PRIVATE lab_common)
set_target_properties(ring_allreduce PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson07-ring-allreduce)
