import torch
from tools.util import get_model
import tools.defaults as defaults
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def export(torch_model_path, onnx_model_path, dynamic_batch=True):
    model = get_model(torch_model_path)
    x_dummy = torch.rand(size=(1, 3, defaults.IMAGE_SIZE, defaults.IMAGE_SIZE)).to(DEVICE)
    input_names = ['input_0']
    output_names = ['output_0']
    if dynamic_batch:
        dynamic_axes = {'input_0': {0: 'batch'}, 'output_0': {0: 'batch'}}
        torch.onnx.export(model, x_dummy, onnx_model_path,
                          input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    else:
        torch.onnx.export(model, x_dummy, onnx_model_path,
                          input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    export(defaults.TORCH_MODEL_PATH, defaults.ONNX_MODEL_PATH)