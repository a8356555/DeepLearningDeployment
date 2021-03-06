from .utils import get_tvm_module_N_params
from .config import cfg

import cv2
import numpy as np
import tvm
from tvm import relay, auto_scheduler, autotvm
from tvm.contrib import graph_executor

from argparse import ArgumentParser

OUTPUT_NUM = 1

def build_inference_vm(mod, params, target, dev):
    #TODO
    # # onnx / keras
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
        intrp = relay.build_module.create_executor("vm", mod, dev, target)    
    return intrp

def build_inference_vm2_for_maskrcnn(mod, params, target, dev):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
        vm_exec = relay.vm.compile(mod, target=target, params=params)
    from tvm.runtime.vm import VirtualMachine
    vm = VirtualMachine(vm_exec, dev)




def build_inference_module_from_auto_tuning(mod, params, graph_opt_sch_file, target, dev):
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
            
    module = runtime.GraphModule(lib["default"](dev))
    return module

def build_inference_module_from_auto_scheduling(mod, params, json_file, target, dev):
    # # pytorch / tf / mxnet / tflite / coreml / darknet / caffe
    with auto_scheduler.ApplyHistoryBest(json_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
    module = graph_executor.GraphModule(lib["default"](dev))
    return module

def _calculate_dhdw_half(h, w):
    """Calculate difference of h or w in order to get a square """
    if h > w:
        dh_half = int(0.1*h/2)
        dw_half = int((h+2*dh_half-w)/2)
    else:
        dw_half = int(0.1*w/2)
        dh_half = int((w+2*dw_half-h)/2)
    return dh_half, dw_half

def preprocess_tvm(image):
    # 加邊框
    h, w, c = image.shape
    np_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dh_half, dw_half = _calculate_dhdw_half(h, w)
    np_img = cv2.copyMakeBorder(np_img, dh_half, dh_half, dw_half, dw_half, cv2.BORDER_REPLICATE)
    np_img = cv2.resize(np_img, (248, 248))[12:236, 12:236]/255.0
    return np_img.transpose(2, 0, 1).astype(cfg.dtype)[np.newaxis, :]


def postprocess_tvm(outputs):
    return outputs.asnumpy()[0]

def vm_output_to_list(o, dtype="float32"):
    if isinstance(o, tvm.nd.NDArray):
        return o.asnumpy()
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.append(vm_output_to_list(f, dtype))
        return result
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))

def tvm_inference(module, img):
    module.set_input(cfg.input_name, tvm.nd.array(img))
    module.run()
    tvm_outputs = []
    for i in range(OUTPUT_NUM):
        tvm_outputs = module.get_output(i)
    return tvm_outputs

def tvm_vm_inference(img, intrp):
    tvm_outputs = intrp.evaluate()(tvm.nd.array(img), **params)
    return tvm_outputs

def make_parser():
    parser = ArgumentParser(
        description=f"Please modify config.py first\nusage ./{__file__} -i /path/to/ur/image -u using_log_or_json_file")
    parser.add_argument(
        '--image-path', '-i', type=str,
        help='using auto-tuning file')
    parser.add_argument(
        '--using-file', '-u', type=str, choices=["log", "json"]
        help='using .log for auto-tuned module, using .json for auto-scheduled one')
    parser..add_argument(
        '--using-vm', '-v', type=str, store_action=False,
        help='using virtual machine (for complicated models)')
    return parser           

if __name__ == "__main__":
    
    parser = make_parser()
    args = parser.parse_args()
    mod, params = get_tvm_module_N_params(    
        model_path=cfg.model_path, 
        input_name=cfg.model_path, 
        batch_size=cfg.batch_size,
        input_shape=cfg.input_shape,
        layout=cfg.layout, 
        dtype=cfg.dtype, 
        use_sparse=cfg.use_sparse
    )
    if args.using_file == "log":
        module = build_inference_module_from_auto_tuning(mod, params, graph_opt_sch_file=cfg.graph_opt_sch_file, target=cfg.target, dev=cfg.dev)
    else: # args.using_file == "json":
        module = build_inference_module_from_auto_scheduling(mod, params, json_file=cfg.json_file, target=cfg.target, dev=cfg.dev):
    
    image = cv2.imread(args.image_path)
    image = preprocess_tvm(image)
    outputs = tvm_inference(image)
    results = postprocess_tvm(outputs)
    print(results)
