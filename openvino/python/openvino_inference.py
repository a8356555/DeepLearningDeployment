import os
import numpy as np
import cv2
import time
from argparse import ArgumentParser
from openvino.inference_engine import IECore, IENetwork

def setup_network(xml_path, bin_path, batch_size=1):
    ie = IECore() #建立推論引擎
    net = IENetwork(model=xml_path, weights=bin_path) #載入模型及權重
    input_blob = next(iter(net.inputs)) #準備輸入空間
    out_blob = next(iter(net.outputs)) #準備輸出空間
    net.batch_size = batch_size #指定批次讀取數量
    n, c, h, w = net.inputs[input_blob].shape #取得批次數量、通道數及影像高、寬
    exec_net = ie.load_network(network=net, device_name="CPU") #載入模型到指定裝置(CPU, GPU, MYRIAD)並產生工作網路
    return exec_net, input_blob, output_blob

def _calculate_dhdw_half(h, w):
    """Calculate difference of h or w in order to get a square """
    if h > w:
        dh_half = int(0.1*h/2)
        dw_half = int((h+2*dh_half-w)/2)
    else:
        dw_half = int(0.1*w/2)
        dh_half = int((w+2*dw_half-h)/2)
    return dh_half, dw_half

def preprocesss_vino(image):
    h, w, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dh_half, dw_half = _calculate_dhdw_half(h, w)
    image = cv2.copyMakeBorder(image, dh_half, dh_half, dw_half, dw_half, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (248, 248))[12:236, 12:236]/255.0
    return image.transpose(2, 0, 1).astype(np.float32)[np.newaxis,:]


def postprocess_vino(vino_outputs):
    return list(vino_outputs.values())[0]

def vino_inference(exec_net, image, input_blob, output_blob):
    res = exec_net.infer(inputs={input_blob: image}) #進行推論，輸出結果陣列大小[1,1000,1,1]
    return res


def make_parser():
    parser = ArgumentParser(
        description=f"usage ./{__file__} -i path/to/ur/image -x path/to/xml/file -b path/to/bin/file [-bs batch_size]")    
    parser.add_argument(
        '--image-path', '-i', type=str, required=True,
    )
    parser.add_argument(
        '--xml-path', '-x', type=str, required=True,
    )
    parser.add_argument(
        '--bin-path', '-b', type=str, required=True
    )
    parser.add_argument(
        '--batch-size', '-bs', type=int, default=1,
    )
    return parser

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()    
    
    exec_net, input_blob, output_blob = setup_network(args.xml_path, args.bin_path, args.batch_size)
    
    image = cv2.imread(args.image_path)
    image = preprocess_vino(image)
    outputs = vino_inference(exec_net, image, input_blob, output_blob)
    resluts = postprocess_onnx(outputs)
    print(resluts)