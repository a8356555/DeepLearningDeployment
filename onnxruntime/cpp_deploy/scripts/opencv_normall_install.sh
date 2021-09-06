#!/bin/bash
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p build && cd build
# Configure
cmake -DCMAKE_INSTALL_PREFIX=/usr/local\ 
-DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules\
-DOPENCV_GENERATE_PKGCONFIG=ON\
../opencv-master
# Build
nproc
make -j8 # cmake --build .
sudo make install
