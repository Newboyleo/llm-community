add_executable(reduce reduce.cu)
target_link_libraries(reduce PRIVATE lab_common)
set_target_properties(reduce PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson05-reduce)
