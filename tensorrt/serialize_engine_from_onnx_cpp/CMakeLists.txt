project(trt_serialize)
add_subdirectory(src)
set_target_properties( trt_serialize
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)