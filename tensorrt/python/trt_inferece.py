import tensorrt as trt
import pycuda.driver as cuda
import common
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_tensorrt_engine_from_onnx(model_path):
    # print('Creating tensorRT Logger...')
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # print('Creating tensorRT Builder...')
    builder = trt.Builder(TRT_LOGGER)
    # print('Creating tensorRT Network...')
    network = builder.create_network(common.EXPLICIT_BATCH)
    # print('Creating tensorRT Config...')
    config = builder.create_builder_config()
    # print('Creating tensorRT Parser...')
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # print('Parsing ONNX Model...')
    config.max_workspace_size = common.GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None
    return builder.build_engine(network, config)

def get_trt_inference_needed_dict(onnx_model_path):
    """
    Return a dict
        {'engine': engine, 'context': context, 'bindings': bindings, 'inputs': inputs, 'outputs': outputs, 'stream': stream}
    """
    # Build a TensorRT engine.
    engine = build_tensorrt_engine_from_onnx(onnx_model_path)

    # # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
    # # Allocate buffers and create a CUDA stream.
    # print('Allocating tensorRT Buffers...')
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    # # Contexts are used to perform inference.
    # print('Creating tensorRT Context...')
    context = engine.create_execution_context()

    return {'engine': engine, 'context': context, 'bindings': bindings, 'inputs': inputs, 'outputs': outputs, 'stream': stream}

def _normalize_test_case(image, pagelocked_buffer, shape=(3, 224, 224), dtype=trt.float32, mean=0.45, std=0.225):
    c, h, w = shape
    image = cv2.resize(image, dsize=(w, h)).transpose([2, 0, 1]).astype(trt.nptype(dtype)).ravel()
    image = (image/255.0 - mean)/std
    np.copyto(pagelocked_buffer, image)
    return image

def trt_inference(
    onnx_model_path=None, 
    trt_dict=None,
    test_image=None,
    preprocess=_normalize_test_case,
    is_test_case_returned=False,
    *args,
    **kwargs):
    """
    Please must pass :
        1) onnx_model_path or 
        2) trt_dict {'engine': engine, 'context': context, 'bindings': bindings, 'inputs': inputs, 'outputs': outputs, 'stream': stream}

    Optional kwargs Input:
        shape: (3, 224, 224) (default)
        dtype: trt.float32 (default)
        mean: 0.45 (default)
        std: 0.225 (default)
    """
    assert onnx_model_path or trt_dict

    if onnx_model_path and trt_dict is None:
        trt_dict = get_trt_inference_needed_dict(onnx_model_path)

    # # Load a normalized test case into the host input page-locked buffer.
    # print('Loading and Normalizing Test Image...')    
    test_case = preprocess(test_image, trt_dict['inputs'][0].host, **kwargs)
    # # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
    # # probability that the image corresponds to that label
    # print('Running Inference...')
    trt_outputs = common.do_inference_v2(trt_dict['context'], bindings=trt_dict['bindings'], inputs=trt_dict['inputs'], outputs=trt_dict['outputs'], stream=trt_dict['stream'])
    # print('done')
    if is_test_case_returned:
        return trt_outputs, test_case
    else:
        return trt_outputs


def preprocess_trt(image, pagelocked_buffer, dtype=trt.float16):
    # 加邊框
    h, w, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dh_half, dw_half = _calculate_dhdw_half(h, w)
    image = cv2.copyMakeBorder(image, dh_half, dh_half, dw_half, dw_half, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (248, 248))[12:236, 12:236].transpose([2, 0, 1]).astype(trt.nptype(dtype)).ravel()/255.0    
    np.copyto(pagelocked_buffer, image)
    return image
    
def postprocess_trt(trt_outputs):
    return trt_outputs[0]



def make_parser():
    parser = ArgumentParser(
        description=f"usage ./{__file__} -i path/to/ur/image -o path/to/ur/onnx/file")    
    parser.add_argument(
        '--image-path', '-i', type=str, default='',
        help='/path/to/ur/image/or/image')
    parser.add_argument(
        '--onnx-path', '-o', type=str, default='',
        help='/path/to/ur/onnx/model/file')
    return parser
    
if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    model = get_model()
    image = cv2.imread(args.image_path)
    trt_dict = get_trt_inference_needed_dict(args.onnx_path)
    trt_outputs = trt_inference(trt_dict=trt_dict, test_image=image, preprocess=preprocess_trt)
    results = postprocess_pt(trt_outputs)
    print(resluts)