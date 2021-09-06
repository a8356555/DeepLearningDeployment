#!/bin/bash
wget https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.8.2.tar.gz
tar zxvf v1.8.2.tar.gz onnxruntime-1.8.2/
cd onnxruntime/
./build.sh --config RelWithDebInfo --build_shared_lib --parallel