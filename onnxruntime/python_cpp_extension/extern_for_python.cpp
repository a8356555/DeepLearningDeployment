// include lib for onnx: /home/luhsuanwen/onnxruntime/include/onnxruntime/core/session
// include lib for other folder src: /home/luhsuanwen/project/DeepLearningModelDeployment/onnxruntime/cpp_deploy/src
// g++ -shared -std=c++11 -Wall -fPIC -o libonnxpy.so /home/luhsuanwen/project/DeepLearningModelDeployment/onnxruntime/cpp_deploy/src/inference.cpp extern_for_python.cpp -I /home/luhsuanwen/onnxruntime/include/onnxruntime/core/session -I /home/luhsuanwen/project/DeepLearningModelDeployment/onnxruntime/cpp_deploy/src -L/home/luhsuanwen/onnxruntime/build/Linux/RelWithDebInfo/ -lonnxruntime -Wl,-R,/home/luhsuanwen/onnxruntime/build/Linux/RelWithDebInfo/ -I /usr/local/include/opencv4/ -L/usr/local/lib -lopencv_calib3d -lopencv_core -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_gapi -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -Wl,-R,/usr/local/lib
// -lopencv_calib3d -lopencv_core -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_gapi -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_videoio -lopencv_aruco -lopencv_barcode -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_datasets -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_face -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform -lopencv_line_descriptor -lopencv_mcc -lopencv_optflow -lopencv_phase_unwrapping -lopencv_plot -lopencv_quality -lopencv_rapid -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_superres -lopencv_surface_matching -lopencv_text -lopencv_tracking -lopencv_videostab -lopencv_wechat_qrcode -lopencv_xfeatures2d -lopencv_ximgproc -lopencv_xobjdetect -lopencv_xphoto

#include "inference.h"
extern "C" float* onnx_inference(char* modelPath, char* imagePath){
	float* output = new float[801];
	predict(modelPath, imagePath, &output);
	return output;
}