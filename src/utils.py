import torch
from model import ResNet18_1D


def load_model(path="../models/resnet18_1d.pt"):
    model = ResNet18_1D()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model