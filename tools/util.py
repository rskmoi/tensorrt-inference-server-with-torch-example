import torch
import torchvision
import tools.defaults as defaults
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_COLORS = defaults.COLORS

def get_model(pretrained_model_path=None):
    """
    This is a sample model for understanding deploying using onnx.
    :param pretrained_model_path:
    :return:
    """
    model = torchvision.models.resnet18(pretrained=False, num_classes=len(_COLORS)).to(DEVICE)
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))

    return model