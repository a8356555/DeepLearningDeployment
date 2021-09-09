#!/bin/bash
cd /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/bin
for MODEL in resnet50 resnet101 efficientnet-b4 efficientnet-b5 efficientnet-b6 efficientnet-b7
do
    if [ -e /content/ONNX_MODELS/$MODEL.trt ]
    then
        echo "$MODEL.trt exists"
    else
        echo $MODEL
        ./trt_serialize /content/ONNX_MODELS/$MODEL.onnx /content/ONNX_MODELS/$MODEL.trt
    fi
done

cd /content/DeepLearningDeployment/tensorrt/cpp_deploy/bin
for MODEL in resnet101 resnet50 efficientnet-b4 efficientnet-b5 efficientnet-b6 efficientnet-b7
do
    echo $MODEL
    ./trt_inference /content/ONNX_MODELS/$MODEL.trt /content/DeepLearningDeployment/sample.jpg $1
done
