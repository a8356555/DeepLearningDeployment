
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <string>
int main() {
    std::string imagePath = "/home/luhsuanwen/project/sample.jpg";
    cv::Mat image = cv::imread(imagePath);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    
    int h = image.rows;
    int w = image.cols;
    int c = image.channels();
    int target_h = 224;
    int target_w = 224;
    int target_c = 3;
    int dh_half, dw_half;
    if (h > w) {
        dh_half = static_cast<int>(0.1*h/2);
        dw_half = static_cast<int>((h+2*dh_half-w)/2);
    } else {
        dw_half = static_cast<int>(0.1*w/2);
        dh_half = static_cast<int>((w+2*dw_half-h)/2);
    }
    
    cv::copyMakeBorder(image, image, dh_half, dh_half, dw_half, dw_half, cv::BORDER_REPLICATE);
    
    std::cout << dh_half << " " << dw_half <<" "<< image.cols << std::endl;

    std::cout << image.at<cv::Vec3b>(0, 0) << std::endl;
    
    cv::resize(image, image, cv::Size(248, 248));
    image.convertTo(image, CV_32FC3, 1.f/255.f);
    
    // HWC to CHW
    cv::Rect ROI(12, 12, target_h, target_w);
    image = image(ROI).clone();
    
    std::cout << image.at<cv::Vec3f>(160, 120) << std::endl;

    std::vector<float> inputTensorValues(target_c*target_h*target_w); 
    assert(image.channels() == target_c);
    for(int _c = 0; _c < target_c; ++_c) {
        for(int _y = 0; _y < target_h; ++_y) {
            for(int _x = 0; _x < target_w; ++_x) {
                inputTensorValues[_c * (target_h * target_w) + _y * target_w + _x] =
                  image.at<cv::Vec3f>(_y, _x)[_c];
            }
        }
    }
    
    for (int i=0; i<224; i++) {
        std::cout << i << " " << inputTensorValues[i] << std::endl;
    }

    return 0;
}