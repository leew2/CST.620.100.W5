import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def gen_img_cap(img_path):
    process = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img = Image.open(img_path).convert("RGB")
    inputs = process(images=img, return_tensors="pt").to(device)

    out = model.generate(**inputs)
    caption = process.decode(out[0], skip_special_tokens=True)

    return caption

