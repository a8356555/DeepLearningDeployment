#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char* argv[])
{
    std::string image_path(argv[1]);
    
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("deploy.so");
    
    // json_graph
    std::ifstream json_in("deploy.json", std::ios::in);
    std::string json_data(std::istreambuf_iterator<char>(json_in), std::istreambuf_iterator<char>());
    json_in.close();
 
    // parameters in binary
    std::ifstream params_in("deploy.params", std::ios::binary);
    std::string params_data(std::istreambuf_iterator<char>(params_in), std::istreambuf_iterator<char>());
    params_in.close();
 
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int dtype_type = kDLCPU;
    int device_id = 0;
 
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
 
    DLTensor* x;

    //Allocatibg memory to the DLTensor object
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

    //Load image file
    Mat rgbaImage = imread(image_path, IMREAD_COLOR);

    //Convert image RGB image from RGBA
    Mat rgbImage;
    rgbaImage.convertTo(rgbImage, CV_32FC3, 2.0/255.0, -1.0);

    Mat bgrImage;
    cvtColor(rgbImage, bgrImage, COLOR_RGB2BGR);

    int h = bgrImage.rows;
    int w = bgrImage.cols;
    int df_half, dw_half;
    if (h > w) {
        dh_half = static_cast<int>(0.1*h/2);
        dw_half = static_cast<int>((h+2*dh_half-w)/2);
    } else {
        dw_half = static_cast<int>(0.1*w/2);
        dh_half = static_cast<int>((w+2*dw_half-h)/2);
    }
    
    // copymakeborder
    cv::copyMakeBorder(bgrImage, bgrImage, dh_half, dh_half, dw_half, dw_half, cv::BORDER_REPLICATE);

    // resize    
    auto target_size = cv::Size(248, 248);
    cv::resize(bgrImage, bgrImage, target_size);
    
    // crop
    cv::Rect ROI(12, 12, 236, 236);
    bgrImage = bgrImage(ROI).clone();

    //Creating xtensor array using opencv
    int in_size = (1 * 64 * 64 * 4 * 3);
    size_t iSize = bgrImage.total();
    size_t ichannels = bgrImage.channels();
    std::vector<int> imgShape = {bgrImage.cols, bgrImage.rows, bgrImage.channels()};
    xt::xarray<float> xArray = xt::adapt((float*)bgrImage.data, iSize * ichannels, xt::no_ownership(), imgShape);

    auto reshapeArray = xt::expand_dims(xArray, 0);

    if(rgbImage.isContinuous())
    {
        //copying image data between cvMatND array to DLTensor object
        TVMArrayCopyFromBytes(x, reshapeArray.data(), in_size);
    }
    else{
        std::cout << " Image array is not continuous... " << std::endl;
        return -1;
    }
    
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("data", x);
 
    tvm::rumtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);
 
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();
 
    DLTensor* y;
    int out_ndim = 1;
    int64_t out_shape[1] = {1000, };
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
 
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);
 
    auto y_iter = static_cast<float*>(y->data);
    auto max_iter = std::max_element(y_iter, y_iter+1000);
    auto max_index = std::distance(y_iter, max_iter);
    std::cout<< "The maximum position in output vector is: " << max_index << std::endl;
 
    TVMArrayFree(x);
    TVMArrayFree(y);
 
    return 0;
}
