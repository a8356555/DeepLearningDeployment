Experimenting on

https://colab.research.google.com/drive/1JXzXEpR6_w5W1doFgUoY5lDl1L7tIAXx

# Table of Contents
* [Deep Learning Deployment](#dld)
    1. [Environmental info](#ei)
    2. [Target](#ta)
    3. [Experiment](#ex)
    4. [TODO](#todo)
    
    
# <a name="dld">1. Deep Learning Deployment

## <a name="ei">Environmental info
    Running on Colab:
        Ubuntu 18.04.5 LTS
        Python 3.7.11
        cuda upgrade to 11.0
        cudnn 7.6.5
        
    Python pytorch related version:
        torch==1.9.0+cu102
        torchvision==0.10.0+cu102
  
  
## <a name="ta">Target
* Be Familiar with ONNX, TensorRT, TVM and other frameworks.   
* Try to speed up using the frameworks above.           
* Deploy models using C++.
  
## <a name="ex">Experiment
1. cpu: onnx vs pytorch
    * There's no difference among outputs of raw pytorch models, static/dynamic onnx models and cpp api models. (e-10)
    * The inference speed of static onnx models is the fastest, at most approximately 2x compared to the one of raw pytorch models.

<img align="center" src="./onnxruntime/pytorch_onnx_inference_speed.png" width="500" height="500">
    
## <a name="todo">TODO
* Openvino (python / cpp api)   
* Tvm cpp api   
* EfficienNet on TVM   
* Mask-RCNN on TensorRT   
* Quantization model
