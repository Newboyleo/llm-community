add_executable(token_routing token_routing.cu)
target_link_libraries(token_routing PRIVATE lab_common)
set_target_properties(token_routing PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson15-token-routing)
