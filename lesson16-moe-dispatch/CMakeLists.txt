add_executable(moe_dispatch moe_dispatch.cu)
target_link_libraries(moe_dispatch PRIVATE lab_common)
set_target_properties(moe_dispatch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson16-moe-dispatch)
