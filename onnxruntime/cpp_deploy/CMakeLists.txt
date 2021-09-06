cmake_minimum_required(VERSION 3.21)
project(onnx_inference)
add_subdirectory(src)
message("path ${CMAKE_CURRENT_SOURCE_DIR}/bin")
message("path ${CMAKE_BINARY_DIR}/bin")

set_target_properties(onnx_inference
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)