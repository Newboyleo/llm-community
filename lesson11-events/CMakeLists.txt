add_executable(events events.cu)
target_link_libraries(events PRIVATE lab_common)
target_compile_options(events PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-default-stream;per-thread>)
set_target_properties(events PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lesson11-events)
