pip install 'pycuda<2021.1'
tar xzvf /content/gdrive/MyDrive/SideProject/nvidia/TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz
cd /content/TensorRT-8.0.1.6/python
pip install tensorrt-8.0.1.6-cp37-none-linux_x86_64.whl
#!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
cd /content/TensorRT-8.0.1.6/graphsurgeon
sudo pip3 install graphsurgeon-0.4.5-py2.py3-none-any.whl
cd /content/TensorRT-8.0.1.6/onnx_graphsurgeon
sudo pip3 install onnx_graphsurgeon-0.3.10-py2.py3-none-any.whl
cp /content/TensorRT-8.0.1.6/targets/x86_64-linux-gnu/lib/lib* /usr/lib