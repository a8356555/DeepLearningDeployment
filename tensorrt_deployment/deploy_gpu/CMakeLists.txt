project(trt_inference)
add_subdirectory(src)
set_target_properties( trt_inference
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)