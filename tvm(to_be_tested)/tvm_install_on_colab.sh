!apt-get -y install \
		llvm \
		clang \
	&& echo "\n\n\nstart installing TVM...\n\n\n" \
	&& apt-get -y install --no-install-recommends \
		gcc	

%cd /content
!git clone --recursive https://github.com/apache/tvm tvm
!sudo apt-get update
!sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
%cd /content/tvm
!mkdir build
!cp cmake/config.cmake build
%cd build
!sed -i 's+USE_CUDA OFF+USE_CUDA /usr/local/cuda/+g' config.cmake
!sed -i 's+USE_LLVM OFF+USE_LLVM ON+g' config.cmake
!cat config.cmake
	
!cmake ..
!make -j4
%cd /content
#TODO 調整文字 import 問題還不確定
!echo 'export TVM_HOME=/content/tvm' >> .bashrc
!echo 'export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}' >> .bashrc
!pip3 install \
		decorator \
		attrs \
		tornado \
		psutil \
		xgboost \
		cloudpickle
!rm -rf /var/lib/apt/lists/*



# sys.path.insert(0, "/content/tvm/python")
