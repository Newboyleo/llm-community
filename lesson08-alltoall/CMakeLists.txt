add_executable(alltoall alltoall.cu)
target_link_libraries(alltoall PRIVATE lab_common)
set_target_properties(alltoall PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson08-alltoall)
