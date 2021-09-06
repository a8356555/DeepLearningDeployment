// https://www.huaweicloud.com/articles/b6b5138e0a59765b0c7df5ec3b7cc084.html
// https://github.com/leimao/ONNX-Runtime-Inference/blob/main/src/inference.cpp
// https://github.com/ZeroE04/CenterNet_onnxruntime
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "inference.h"

using namespace std;
template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

void preProcessImage(const std::string& imagePath, std::vector<float>& inputTensorValues) {
    cv::Mat image = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Input image: " << imagePath << " load failed\n";
        return;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int h = image.rows;
    int w = image.cols;
    int t_c = 3; // target c
    int t_h = 224;
    int t_w = 224;
    int dh_half, dw_half;
    if (h > w) {
        dh_half = static_cast<int>(0.1*h/2);
        dw_half = static_cast<int>((h+2*dh_half-w)/2);
    } else {
        dw_half = static_cast<int>(0.1*w/2);
        dh_half = static_cast<int>((w+2*dw_half-h)/2);
    }
    cv::copyMakeBorder(image, image, dh_half, dh_half, dw_half, dw_half, cv::BORDER_REPLICATE);
    cv::resize(image, image, cv::Size(248, 248));
    image.convertTo(image, CV_32FC3, 1.f/255.f);
    
    // HWC to CHW
    cv::Rect ROI(12, 12, t_h, t_w);
    image = image(ROI).clone();
    
    // // Normalization per channel
    // // Normalization parameters obtained from
    // cv::Mat channels[3];
    // cv::split(image, channels);    
    // channels[0] = (channels[0] - 0.485) / 0.229;
    // channels[1] = (channels[1] - 0.456) / 0.224;
    // channels[2] = (channels[2] - 0.406) / 0.225;
    // cv::merge(channels, 3, image);  

    // HWC to CHW
    assert(image.channels() == t_c);
    for(int c = 0; c < t_c; ++c) {
        for(int y = 0; y < t_h; ++y) {
            for(int x = 0; x < t_w; ++x) {
                inputTensorValues[c * (t_h * t_w) + y * t_w + x] =
                  image.at<cv::Vec3f>(y, x)[c];
            }
        }
    }    
    return;
}

void postProcessOutput(float* p_output[], std::vector<Ort::Value>& outputTensors) {
    float* output = outputTensors[0].GetTensorMutableData<float>();        
    for (int i = 0; i < 801; ++i) {
        (*p_output)[i] = output[i];
    }      
    return;
}

void predict(const std::string& modelPath, const std::string& imagePath, float* p_output[], unsigned int test_num) {
    std::string instanceName = "image-classification-inference";
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                instanceName.c_str());    
    Ort::SessionOptions sessionOptions;
    // 使用一个线程执行op
    sessionOptions.SetIntraOpNumThreads(1);
    
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    Ort::Session session(env, modelPath.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;

    // size_t numInputNodes = session.GetInputCount();
    // size_t numOutputNodes = session.GetOutputCount();

    // std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    // std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    const char* inputName = session.GetInputName(0, allocator);
    // std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    // ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    // std::cout << "Input Type: " << inputType << std::endl;
    
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    // std::cout << "Input Dimensions: " << inputDims << std::endl;

    const char* outputName = session.GetOutputName(0, allocator);
    // std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    // ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    // std::cout << "Output Type: " << outputType << std::endl;
    
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    // std::cout << "Output Dimensions: " << outputDims << std::endl;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};  

    for (int i=0; i<test_num; i++) {
        std::vector<Ort::Value> inputTensors;
        std::vector<Ort::Value> outputTensors;
        size_t inputTensorSize = vectorProduct(inputDims);
        std::vector<float> inputTensorValues(inputTensorSize);
        preProcessImage(imagePath, inputTensorValues);          

        size_t outputTensorSize = vectorProduct(outputDims);
        std::vector<float> outputTensorValues(outputTensorSize);      

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
            inputDims.size()));

        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValues.data(), outputTensorSize,
            outputDims.data(), outputDims.size()));

        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);
                
        postProcessOutput(p_output, outputTensors);
    }   
    auto t_end = std::chrono::high_resolution_clock::now();
    auto elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count()/test_num;
    std::cout << "time cost: " << elapsed_time_ms << "ms" << std::endl;
    return;
}