#include <string>
#include "inference.h"

int main(int argc, char** argv) {    
    std::string modelPath = "/home/luhsuanwen/project/ONNX_MODELS/efficientnet-b4.onnx";    
    std::string imagePath = "/home/luhsuanwen/project/sample.jpg";
    unsigned int test_num = 2;
    if (argc > 1) {
        modelPath = argv[1];
    }
    if (argc > 2) {
        imagePath = argv[2];
    }
    if (argc > 3) {
        test_num = atoi(argv[3]);
    }
    std::cout << modelPath << std::endl;
    float* output = new float[801];
    predict(modelPath, imagePath, &output, test_num);
    return 0;
}