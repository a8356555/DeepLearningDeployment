Experimenting GPU environment on:

https://colab.research.google.com/drive/1JXzXEpR6_w5W1doFgUoY5lDl1L7tIAXx

Experimenting CPU environment on Google Cloud Platform (see jupyter notebook in repo)

# Table of Contents
* [Deep Learning Deployment](#dld)
    1. [Environmental info](#ei)
    2. [Target](#ta)
    3. [Experiment](#ex)
    4. [TODO](#todo)
    5. [NOTE](#note)
    
    
# <a name="dld">1. Deep Learning Deployment

## <a name="ei">Environmental info
* Cpu experiment environment
    
        - Ubuntu 18.04.5 LTS
        - Cpu 4 cores Intel(R) Xeon(R) CPU @ 2.30GHz
        - RAM 16 GB DIMM RAM Synchronous
        
        - Python 3.6.9
        - torch 1.9.0+cu102
        - torchvision 0.10.0+cu102
        - OpenCV 4.5.2
        - Onnxruntime v1.8.2
        - TVM 0.8
        - OpenVINO 2021.1.110
        
* Gpu experiment environment 
    
        - Ubuntu 18.04.5 LTS
        - Cpu 2 cores Intel(R) Xeon(R) CPU @ 2.30GHz
        - RAM 12GB
        - GPU Tesla K80    
    
        - Python 3.7.11    
        - CUDA 11.2
        - Cudnn 7.6.5                
        - torch 1.9.0+cu102
        - torchvision 0.10.0+cu102
        - OpenCV 4.5.2 built with CUDA
        - TensorRT 8.0.1.6
    
## <a name="ta">Target
* Have an overall understanding of multiple frameworks used for speeding up deployment and get familiar with their structure. eg. ONNX Runtime, TensorRT, TVM, Openvino, etc.
* Compare the inference speed of frameworks above.
* Try to deploy models using C++ API.
* Try to use ctypes to call c++ implemented inference function.
* Modify preprocessing step to make sure different frameworks may have the same prediction.
    
## <a name="ex">Experiment
1. CPU: Pytorch vs ONNX Runtime vs TVM vs OpenVINO
    * Outputs from different frameworks are mostly the same. (mse=e-10)
    * TVM is slower than Pytorch, probably because the parameter num_measure_trials is too small.
    * ONNX Runtime is approximately 1.3x ~ 2x faster than Pytorch.
    * ONNX Runtime on C++ is slower than on python, probably not the best implementation or too much overhead.
    * ONNX Runtime using ctypes to call c++ implemented function is much more slower, probably because of too much calling overhead.
    * OpenVINO is 2x ~ 3x faster than Pyotrch, the best approach so far.

<p align="center">
    <img src="./onnxruntime/pytorch_onnx_inference_speed.png" width="500" height="500">
</p><br>    
    
2. GPU: Pytorch vs ONNX Runtime vs TensorRT vs TVM
    * Outputs of Pytorch, ONNX Runtime and TVM are mostly the same. (mse=e-10)
        * Output of TensorRT vs Pytorch are slightly different. (mse=e-4)
    * TensorRT on C++ is slower
    
    
## <a name="todo">TODO
* CPU/GPU/MaskRCNN TVM with bigger tuning option parameter num_measure_trials = 800*len(tasks)
* Tvm CPP API
* Mask-RCNN on TensorRT / OpenVINO
* Check why Mask-RCNN result differs.
* Best configuration on different framework
* Mixed Precision model
    
## <a name="note">NOTE
* cv::cuda::resize has different results compared to cv::resize (the former always use the INTER_NEAREST flat no matter what you pass)
    https://github.com/opencv/opencv/issues/4728
* Different models may need different package version to support, eg. <br>
    Mask-RCNN: pytorch to tvm: torch 1.7.0 + torchvision 0.8.1<br>
    Efficientnet: pytorch to onnx: torch 1.9.0, opset_level=10 (11 failed)
