#!/bin/bash
cd /opt/intel/openvino_2021/deployment_tools/model_optimizer
for MODEL in resnet101 resnet50 efficientnet-b4 efficientnet-b5 efficientnet-b6 efficientnet-b7
do
   python3 mo.py --input_model /home/luhsuanwen/project/ONNX_MODELS/$MODEL.onnx --output_dir /home/luhsuanwen/project/VINO_MODELS
done