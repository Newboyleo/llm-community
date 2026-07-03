add_executable(broadcast broadcast.cu)
target_link_libraries(broadcast PRIVATE lab_common)
set_target_properties(broadcast PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson03-broadcast)
