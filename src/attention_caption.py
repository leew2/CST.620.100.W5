from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image


def get_attention_caption(img):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processors = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    img = Image.open(img).convert("RGB")
    inputs = processors(images=img, return_tensors="pt")
    caption_id = model.generate(**inputs)

    caption = tokenizer.decode(caption_id[0], skip_special_tokens=True)
    return caption
