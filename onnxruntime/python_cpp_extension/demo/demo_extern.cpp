// include lib for onnx: /home/luhsuanwen/onnxruntime/include/onnxruntime/core/session
// include lib for other folder src: 
// g++ -Wall -fPIC -c demo_predict.cpp -o demo_predict.o -I /home/luhsuanwen/onnxruntime/include/onnxruntime/core/session -L/home/luhsuanwen/onnxruntime/build/Linux/RelWithDebInfo/ -lonnxruntime -Wl,-R,/home/luhsuanwen/onnxruntime/build/Linux/RelWithDebInfo/
// g++ -Wall -fPIC -c demo_extern.cpp -o demo_extern.o
// 從.o to .so: g++ -shared -Wall -fPIC -o libdemoonnxpy.so *.o 
// 直接從 src file: g++ -shared -Wall -fPIC -o libdemoonnxpy.so demo_extern.cpp 

// g++ -shared -std=c++11 -Wall -fPIC -o libdemoonnxpy.so demo_predict.cpp demo_extern.cpp -I /home/luhsuanwen/onnxruntime/include/onnxruntime/core/session -L/home/luhsuanwen/onnxruntime/build/Linux/RelWithDebInfo/ -lonnxruntime -Wl,-R,/home/luhsuanwen/onnxruntime/build/Linux/RelWithDebInfo/

#include "demo_predict.h"

extern "C" float* onnx_inference(char *modelPath, char *imagePath){		//char *onnx_path, char *image_path    
    return predict(modelPath, imagePath);
}

