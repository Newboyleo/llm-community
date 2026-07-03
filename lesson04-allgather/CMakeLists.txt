add_executable(allgather allgather.cu)
target_link_libraries(allgather PRIVATE lab_common)
set_target_properties(allgather PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson04-allgather)
