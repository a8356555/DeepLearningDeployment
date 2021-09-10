import os
import numpy as np
import onnx

import tvm
from tvm import relay
import tvm.contrib.graph_executor as runtime

def get_network_from_onnx(
    model_path, 
    input_name="input.1", 
    input_shape=(1, 3, 224, 224), 
    dtype="float32"
):
    onnx_model = onnx.load(model_path)
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict, dtype=dtype)
    return mod, params
    
def convert_layout(mod):
    desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod

def get_tvm_module_N_params(
    model_path, 
    input_name="input.1", 
    batch_size=1,
    input_shape=(3, 224, 224),
    layout="NHWC", 
    dtype="float32", 
    use_sparse=False
):    
"""Get the symbol definition and random weight of a network"""
    data_shape = (batch_size,) + input_shape

    mod, params = get_network_from_onnx(model_path, input_name, data_shape, dtype)
    if layout == "NHWC":
        mod = convert_layout(mod)
    # net = mod["main"]
    # net = relay.Function(
    #     net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
    # )
    # mod = tvm.IRModule.from_expr(net)
    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse
        mod, params = convert_model_dense_to_sparse(mod, params, bs_r=4, random_params=True)
    
    return mod, params
