import torch
import numpy as np
import cv2
from argparse import ArgumentParser


def _calculate_dhdw_half(h, w):
    """Calculate difference of h or w in order to get a square """
    if h > w:
        dh_half = int(0.1*h/2)
        dw_half = int((h+2*dh_half-w)/2)
    else:
        dw_half = int(0.1*w/2)
        dh_half = int((w+2*dw_half-h)/2)
    return dh_half, dw_half


def preprocess_pt(image):
    # 加邊框
    h, w, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dh_half, dw_half = _calculate_dhdw_half(h, w)
    image = cv2.copyMakeBorder(image, dh_half, dh_half, dw_half, dw_half, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (248, 248))[12:236, 12:236]
    tensor = torch.tensor(image, dtype=torch.float)    
    return tensor.permute(2, 0, 1).div(255.0).unsqueeze(0)

def postprocess_pt(pt_outputs):
    return pt_outputs.detach().to(device).numpy()[0]

def get_model():
    pass


def make_parser():
    parser = ArgumentParser(
        description=f"usage {__file__} -i path/to/ur/image")    
    parser.add_argument(
        '--image-path', '-i', type=str, default='',
        help='/path/to/ur/image/or/image')
    return parser
    
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = make_parser()
    args = parser.parse_args()
    model = get_model()
    model.to(device)
    model.eval()
    image = cv2.imread(args.image_path)
    tensor = preoprocess_pt(image)
    output = model(tensor.to(device))
    results = postprocess_pt(output)
    print(resluts)