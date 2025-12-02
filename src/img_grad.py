import torch.nn as nn
import torchvision.models as models
import plotlib.pyplot as plt

def get_img_grad(img):
    model = models.resnet18(pretrained=True)
    model.eval()

    img.requires_grad_(True)
    output = model(img.unsqueeze(0))
    class_idx = torch.argmax(output, dim=1)
    output[0, class_idx].backward()

    saliency = img.grad.data.abs().squeeze().cpu()
    return saliency

