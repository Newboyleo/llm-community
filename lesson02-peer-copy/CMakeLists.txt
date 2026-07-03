add_executable(peer_copy peer_copy.cu)
target_link_libraries(peer_copy PRIVATE lab_common)
set_target_properties(peer_copy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson02-peer-copy)
