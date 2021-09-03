#!/bin/bash
cd /content/DeepLearningDeployment/tensorrt/serialize_engine_from_onnx_cpp/bin
for MODEL in resnet101 resnet50 efficientnet-b4 efficientnet-b5 efficientnet-b6 efficientnet-b7
do
    echo $MODEL
    ./trt_serialize /content/$MODEL.onnx /content/$MODEL.trt
done

cd /content/DeepLearningDeployment/tensorrt/cpp_deploy/bin
for MODEL in resnet101 resnet50 efficientnet-b4 efficientnet-b5 efficientnet-b6 efficientnet-b7
do
    echo $MODEL
    ./trt_inference /content/$MODEL.trt /content/YuShanAICompetition/sample.jpg $1
done