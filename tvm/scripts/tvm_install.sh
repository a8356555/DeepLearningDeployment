apt-get update \
	&& apt-get -y upgrade \
	&& apt-get -y install --no-install-recommends \
		vim \
		python3-pip \
		git \
	&& apt-get -y install \
		llvm \
		clang \
	&& echo "\n\n\nstart installing TVM...\n\n\n" \
	&& git clone --recursive https://github.com/apache/tvm tvm \
	&& apt-get -y install --no-install-recommends \
		python3-dev \
		python3-setuptools \
		gcc \
		libtinfo-dev \
		zlib1g-dev \
		build-essential \
		libedit-dev \
		libxml2-dev \
	&& cd ~/tvm \
	&& mkdir build \
	&& cp cmake/config.cmake build \
	&& cd build \
	&& sed -i 's+USE_CUDA OFF+USE_CUDA /usr/local/cuda/+g' config.cmake \
	&& sed -i 's+USE_LLVM OFF+USE_LLVM ON+g' config.cmake \
	&& cmake .. \
	&& make -j4 \
	&& cd ~ \
	&& echo 'export TVM_HOME=/root/tvm' >> .bashrc \
	&& echo 'export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}' >> .bashrc \
	&& pip3 install \
		decorator \
		attrs \
		tornado \
		psutil \
		xgboost \
	
