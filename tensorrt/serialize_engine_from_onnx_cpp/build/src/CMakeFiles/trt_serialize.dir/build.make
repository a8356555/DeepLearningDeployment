# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python2.7/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python2.7/dist-packages/cmake/data/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build

# Include any dependencies generated for this target.
include src/CMakeFiles/trt_serialize.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/trt_serialize.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/trt_serialize.dir/flags.make

src/CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.o: src/CMakeFiles/trt_serialize.dir/flags.make
src/CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.o: ../src/trt_parse_onnx_N_save.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.o"
	cd /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.o -c /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/src/trt_parse_onnx_N_save.cpp

src/CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.i"
	cd /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/src/trt_parse_onnx_N_save.cpp > CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.i

src/CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.s"
	cd /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/src/trt_parse_onnx_N_save.cpp -o CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.s

# Object files for target trt_serialize
trt_serialize_OBJECTS = \
"CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.o"

# External object files for target trt_serialize
trt_serialize_EXTERNAL_OBJECTS =

../bin/trt_serialize: src/CMakeFiles/trt_serialize.dir/trt_parse_onnx_N_save.cpp.o
../bin/trt_serialize: src/CMakeFiles/trt_serialize.dir/build.make
../bin/trt_serialize: /content/TensorRT-8.0.1.6/lib/libnvcaffe_parser.so
../bin/trt_serialize: /content/TensorRT-8.0.1.6/lib/libnvinfer.so
../bin/trt_serialize: /content/TensorRT-8.0.1.6/lib/libnvinfer_plugin.so
../bin/trt_serialize: /content/TensorRT-8.0.1.6/lib/libnvonnxparser.so
../bin/trt_serialize: /content/TensorRT-8.0.1.6/lib/libnvparsers.so
../bin/trt_serialize: src/CMakeFiles/trt_serialize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/trt_serialize"
	cd /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/trt_serialize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/trt_serialize.dir/build: ../bin/trt_serialize

.PHONY : src/CMakeFiles/trt_serialize.dir/build

src/CMakeFiles/trt_serialize.dir/clean:
	cd /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build/src && $(CMAKE_COMMAND) -P CMakeFiles/trt_serialize.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/trt_serialize.dir/clean

src/CMakeFiles/trt_serialize.dir/depend:
	cd /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/src /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build/src /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/build/src/CMakeFiles/trt_serialize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/trt_serialize.dir/depend

