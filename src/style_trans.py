import torch.optim as optim
import torch
import torchvision.models as model
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def get_style_transfer(img, style):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = Image.open(img)
    style_img = Image.open(style)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    content_img = preprocess(content_img).unsqueeze(0).to(device)
    style_img = preprocess(style_img).unsqueeze(0).to(device)
    vgg = model.vgg19(pretrained=True).features.eval().to(device)
    optimizer = optim.Adam([content_img], lr=0.01)

    tar = vgg(content_img) # checkin g tensor differences
    print(f'IMG:{content_img.shape}, STYLE:{style_img.shape}, VGG{tar.shape}')
    

    for step in range(500):
        target_content = vgg(content_img)
        target_style = vgg(style_img)
        loss = content_loss(target=target_content, content=content_img) + 1e6 * style_loss(target=target_style, style=style_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    plt.imshow(img.squeeze().permute(1, 2, 0).detach().numpy())
    plt.axis('off')
    plt.show()


def gram_matrix(feature_map):
    _, c, h, w = feature_map.size()
    features = feature_map.view(c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)

def style_loss(target, style):
    return torch.mean((gram_matrix(target) - gram_matrix(style)) ** 2)

def content_loss(target, content):
    return torch.mean((target - content) ** 2)