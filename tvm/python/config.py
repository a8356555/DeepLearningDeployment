import tvm
import os

class cfg:
    model_name = "resnet101"
    model_path = f"ONNX_MODELS/{model_name}.onnx"
    input_name = "input.1"
    use_sparse = False
    batch_size = 1
    input_shape = (3, 224, 224)
    output_shape = (batch_size, 1000)
    dtype = "float32"
    layout = "NCHW"
    opset_version = 11
    
# # Define the neural network and compilation target.
# # If the target machine supports avx512 instructions, replace the
# # "llvm -mcpu=core-avx2" with "llvm -mcpu=skylake-avx512"
#     target = tvm.target.Target("llvm -mcpu=core-avx2")
    target = "llvm"
    json_file = "TVM_FILES/%s-%s-B%d-%s.json" % (model_name, layout, batch_size, target.kind.name)
    log_file = f'TVM_FILES/{model_name}.log'
    graph_opt_sch_file = f'TVM_FILES/{model_name}_graph_opt.log'
#     Set number of threads used for tuning based on the number ofphysical CPU cores on your machine.
    num_threads = 1
    dev = tvm.cpu()

os.environ["TVM_NUM_THREADS"] = str(cfg.num_threads)
