cmake_minimum_required(VERSION 3.12)
project(onnx_inference)

# list(APPEND CMAKE_PREFIX_PATH "/home/luhsuanwen/opencv/build")

find_package(OpenCV REQUIRED)
message("opencv ${OpenCV_LIBS}")
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /home/luhsuanwen/onnxruntime/include/onnxruntime/core/session)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /home/luhsuanwen/onnxruntime/build/Linux/RelWithDebInfo)

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
link_directories(${OpenCV_LIBS})
add_executable(onnx_inference inference.cpp main.cpp)

target_include_directories(onnx_inference PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnx_inference PUBLIC ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS   "-g")                     # 调试信息
set(CMAKE_CXX_FLAGS   "-Wall")                  # 开启所有警告