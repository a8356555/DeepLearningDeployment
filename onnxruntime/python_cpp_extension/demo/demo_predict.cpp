#include <string>
#include <iostream>
#include "demo_predict.h"
#include "onnxruntime_cxx_api.h"

float* predict(const std::string& modelPath, const std::string& imagePath, unsigned int test_num) {
    std::string instanceName = "image-classification-inference";
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                instanceName.c_str()); 
    
    std::cout << modelPath << " " << imagePath << std::endl;
    

    static float arr[5] = {1.7, 1.2, 1.3, 1.4, 1.5};
    return arr;
}
