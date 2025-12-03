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
    # get image
    content_img = preprocess(content_img).unsqueeze(0).to(device)
    style_img = preprocess(style_img).unsqueeze(0).to(device)
    vgg = model.vgg19(pretrained=True).features.eval().to(device)
    
    # get features
    target_feature = vgg(content_img).detach()
    target_style = vgg(style_img).detach()
    
    gen_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([gen_img], lr=0.01)

    print(f'IMG:{content_img.shape}, STYLE:{style_img.shape}, target{target_feature.shape}')
    amount = 4000
    for step in range(amount):
        pred_img = vgg(gen_img)
        loss = content_loss(target=target_feature, content=pred_img) + 1e6 * style_loss(target=target_style, style=pred_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            percent = (step / amount) * 100
            print(f'Step [{step}/{amount}] ({percent:.2f}%), Loss: {loss.item():.4f}')
    new_img = gen_img.cpu().clone().squeeze(0)
    plt.subplot(1,3,1)
    plt.imshow(content_img.permute(0, 2, 3, 1).squeeze(0).cpu().numpy())
    plt.title("Base")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(style_img.permute(0, 2, 3, 1).squeeze(0).cpu().numpy())
    plt.title("Style")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(new_img.permute(1, 2, 0).detach().numpy())
    plt.title("Generated")
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