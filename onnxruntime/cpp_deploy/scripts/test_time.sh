#!/bin/bash
cd /home/luhsuanwen/project/DeepLearningModelDeployment/onnxruntime/cpp_deploy/bin
for MODEL in resnet50 resnet101 efficientnet-b4 efficientnet-b5 efficientnet-b6 efficientnet-b7
do
    echo $MODEL    
    ./onnx_inference /home/luhsuanwen/project/ONNX_MODELS/$MODEL.onnx /home/luhsuanwen/project/sample.jpg $1
done