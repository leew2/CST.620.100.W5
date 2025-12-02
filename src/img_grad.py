import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def get_img_grad(img):
    model = models.resnet18(pretrained=True)
    model.eval()

    img = Image.open(img).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = preprocess(img)
    img.requires_grad = True

    output = model(img.unsqueeze(0))
    class_idx = torch.argmax(output)
    output[0, class_idx].backward()
    saliency = img.grad.abs().squeeze().permute(1, 2, 0).numpy()

    plt.imshow(saliency, cmap='hot')
    plt.axis('off')
    plt.show()


