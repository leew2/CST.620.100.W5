from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

def get_attention_caption(img):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processors = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    inputs = processors(images=img, return_tensors="pt")
    caption_id = model.generate(**inputs)

    caption = tokenizer.decode(caption_id[0], skip_special_tokens=True)
    return caption
