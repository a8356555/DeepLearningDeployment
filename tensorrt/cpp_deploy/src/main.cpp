#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "inference.h"

int main(int argc, char* argv[])
{
    int test_num = 10;
    if (argc<3)
    {
        std::cerr << "basic usage: " << argv[0] << " engine.trt image.jpg\n" << "test speed usage: " << argv[0] << " engine.trt image.jpg test_loop_number" << std::endl;        
        return -1;
    } else {
        std::string engine_path(argv[1]);
        std::string image_path(argv[2]);
        float* cpu_output = new float[801];
        // cpu_result = predict(engine_path, image_path);
        if (argc == 4)
            test_num = atoi(argv[3]);
        evaluate_predict_speed(engine_path, image_path, &cpu_output, test_num);
    }        
    return 0;
}