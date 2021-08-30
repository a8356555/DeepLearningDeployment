import onnx
import torch

def get_and_check_onnx_model(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print('if nothing showed, then the model is fine.')
    return onnx_model


def pytorch_to_onnx(model, model_path, input_shape=(1, 3, 224, 224), opset_version=11):        
    input_data = torch.randn(input_shape)
    torch.onnx.export(model.eval(), input_data, model_path, opset_version=opset_version)
    print('Export Finished, now Checking ONNX Model')    
    get_and_check_onnx_model(model_path)


def set_efficient_model_ready(model):
    model.set_swish(memory_efficient=False)
    return model


def make_parser():
    parser = ArgumentParser(
        description=f"usage {__file__} -n model_name -o path/to/ur/output/onnx/file")    
    parser.add_argument(
        '--model-name', '-n', type=str, default='',
        help='')
    parser.add_argument(
        '--output-path', '-o', type=str, default='',
        help='/path/to/ur/output/onnx/file')
    return parser
    
def get_model(args):
    if "eff" in args.model_name:
        model = "..."
        set_efficient_model_ready(model)
    return model

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    model = get_model(args)
    pytorch_to_onnx(model, args.output_path)
    print(resluts)