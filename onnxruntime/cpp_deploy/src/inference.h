#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

template <typename T>
T vectorProduct(const std::vector<T>& v);

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v);

std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type);

void preProcessImage(const std::string& imagePath, std::vector<float>& inputTensorValues);

void postProcessOutput(float* p_output[], std::vector<Ort::Value>& outputTensors);

void predict(const std::string& modelPath, const std::string& imagePath, float* p_output[], unsigned int test_num=1);
