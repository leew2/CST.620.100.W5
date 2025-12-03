import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def get_img_grad(img, attention_caption=None, img_caption=None):
    model = models.resnet18(pretrained=True)
    model.eval()

    img1 = Image.open(img).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = preprocess(img1)
    img.requires_grad = True

    output = model(img.unsqueeze(0))
    class_idx = torch.argmax(output)
    output[0, class_idx].backward()
    saliency = img.grad.abs().squeeze().permute(1, 2, 0).numpy()

    
    plt.title(f'Attention Caption: {attention_caption}\nImage Caption: {img_caption}')
    plt.subplot(1,2,1)
    plt.imshow(saliency, cmap='hot')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img1.resize((224,224)))
    plt.axis('off')
    plt.show()


