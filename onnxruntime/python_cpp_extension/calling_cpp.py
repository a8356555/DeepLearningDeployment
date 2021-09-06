import ctypes
import os
model_path = "/home/luhsuanwen/project/ONNX_MODELS/resnet50.onnx"
image_path = "/home/luhsuanwen/project/sample.jpg"
modelPath = model_path.encode(encoding="utf-8")
imagePath = image_path.encode(encoding="utf-8")

so_path = "/home/luhsuanwen/project/DeepLearningModelDeployment/onnxruntime/python_cpp_extension/libonnxpy.so"
dll = ctypes.cdll.LoadLibrary(so_path)
dll.onnx_inference.restype = ctypes.POINTER(ctypes.c_float)
dll.onnx_inference.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
n = 10
import time
s = time.time()
for _ in range(n):
    output_arr = dll.onnx_inference(modelPath, imagePath)
e = time.time()
print("time cost: ", (e-s)/n)