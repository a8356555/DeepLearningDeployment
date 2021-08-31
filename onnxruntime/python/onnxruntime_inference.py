# install onnx / onnxruntim / efficientnet first

import onnxruntime
import os
import numpy as np
import cv2
import time


    
def get_and_check_onnx_sesson(model_path):
    import onnx
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print('if nothing showed, then the model is fine.')
    return onnx_model


def _calculate_dhdw_half(h, w):
    """Calculate difference of h or w in order to get a square """
    if h > w:
        dh_half = int(0.1*h/2)
        dw_half = int((h+2*dh_half-w)/2)
    else:
        dw_half = int(0.1*w/2)
        dh_half = int((w+2*dw_half-h)/2)
    return dh_half, dw_half
    
def preprocess_onnx(image):
    # 加邊框
    h, w, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dh_half, dw_half = _calculate_dhdw_half(h, w)
    image = cv2.copyMakeBorder(image, dh_half, dh_half, dw_half, dw_half, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (248, 248))[12:236, 12:236]/255.0
    image = image.transpose(2, 0, 1).astype(np.float32)[np.newaxis,:]
    return image

def show_onnx_session_io_name(session):
    session.get_modelmeta()
    first_input_name = session.get_inputs()[0].name
    first_output_name = session.get_outputs()[0].name
    print(path, first_input_name, first_output_name)
    
def onnxruntime_inference(image, ort_session):
    image = preprocess_onnx(image)
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

def postprocess_onnx(onnx_outputs):
    return onnx_outputs[0][0]


def make_parser():
    parser = ArgumentParser(
        description=f"usage ./{__file__} -i path/to/ur/image -p path/to/onnx/model/file")    
    parser.add_argument(
        '--image-path', '-i', type=str,
    )
    parser.add_argument(
        '--onnx-path', '-o', type=str,
    )
    return parser

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    session = get_and_check_onnx_session(args.onnx_path)
    image = cv2.imread(args.image_path)
    inp = preprocess_onnx(image)
    outputs = onnxruntime_inference(inp, session)
    resluts = postprocess_onnx(outputs)
    print(resluts)