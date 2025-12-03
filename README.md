# Image Processing and Captioning Project

A comprehensive Python project that combines multiple deep learning techniques for image analysis and generation, including image captioning, attention mechanisms, gradient visualization, and neural style transfer.

## Features

- **Image Captioning**: Generate descriptive captions for images using two different models
  - BLIP (Bootstrapping Language-Image Pre-training) for base captioning
  - ViT-GPT2 with attention mechanism for enhanced captions
  
- **Gradient Visualization**: Visualize image saliency maps using ResNet-18 to understand which parts of the image contribute most to the model's predictions

- **Neural Style Transfer**: Apply artistic style transfer using VGG-19 to generate images that combine the content of one image with the style of another

## Project Structure

```
.
├── img/                          # Image directory for input images
│   ├── cat.jpg                  # Example content image
│   └── style.webp               # Example style image
├── src/                         # Source code directory
│   ├── main.py                  # Main entry point
│   ├── img_caption.py           # BLIP-based image captioning
│   ├── attention_caption.py     # Attention-based image captioning
│   ├── img_grad.py              # Gradient/saliency visualization
│   └── style_trans.py           # Neural style transfer
└── README.md                    # This file
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional but recommended for faster processing)

See `requirements.txt` for Python package dependencies.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CST.620.100.W5
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

Execute all image processing tasks:
```bash
python src/main.py
```

This will:
1. Generate two image captions (BLIP and attention-based)
2. Create a gradient/saliency visualization
3. Perform neural style transfer

### Using Individual Modules

#### Image Captioning
```python
from src.img_caption import gen_img_cap
from src.attention_caption import get_attention_caption

# BLIP captioning
caption = gen_img_cap("path/to/image.jpg")

# Attention-based captioning
caption = get_attention_caption("path/to/image.jpg")
```

#### Gradient Visualization
```python
from src.img_grad import get_img_grad

get_img_grad(img="path/to/image.jpg", 
             attention_caption="sample caption",
             img_caption="sample caption")
```

#### Style Transfer
```python
from src.style_trans import get_style_transfer

get_style_transfer(content_img="path/to/content.jpg", 
                   style_img="path/to/style.jpg")
```

## Model Details

### BLIP (Image Captioning)
- Model: `Salesforce/blip-image-captioning-base`
- Purpose: Generate concise image descriptions
- Device: Automatically uses GPU if available

### ViT-GPT2 (Attention Captioning)
- Model: `nlpconnect/vit-gpt2-image-captioning`
- Purpose: Generate attention-weighted image captions
- Architecture: Vision Transformer + GPT-2

### ResNet-18 (Gradient Visualization)
- Model: ImageNet pre-trained ResNet-18
- Purpose: Compute saliency maps to visualize important image regions

### VGG-19 (Style Transfer)
- Model: ImageNet pre-trained VGG-19
- Purpose: Extract content and style features for neural style transfer
- Optimization: Uses Adam optimizer with 4000 training steps

## Configuration

Update the following in `src/main.py` to use different images:
```python
content = "img/cat.jpg"      # Path to content image
style = "img/style.webp"     # Path to style image
```

## Output

- **Captions**: Printed to console
- **Saliency Map**: Displayed side-by-side with original image
- **Style Transfer**: Displayed comparison of content, style, and generated images

## Requirements

See `requirements.txt` for complete list of dependencies.

## Notes

- Models are automatically downloaded on first use (~2-3 GB total)
- GPU recommended for faster processing (4000 style transfer iterations can take time on CPU)
- Image size is standardized to 224x224 for model compatibility
- Requires active internet connection for model downloads

## Author

CST.620.100.W5 Course Project

## License

This project is for educational purposes.
