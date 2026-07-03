add_executable(reduce_scatter reduce_scatter.cu)
target_link_libraries(reduce_scatter PRIVATE lab_common)
set_target_properties(reduce_scatter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson06-reduce-scatter)
