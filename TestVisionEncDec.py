from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, GPT2TokenizerFast, VisionEncoderDecoderModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import torch 

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

import urllib.parse as parse
import os

# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
        

# a function to perform inference
def get_caption(model, image_processor, tokenizer, image_path):
    
    image = load_image(image_path)
    # preprocess the image
    img = image_processor(image, return_tensors="pt").to(device)
    # generate the caption (using greedy decoding by default)
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
 

    attention_maps = output.attentions  # Get attention maps

    # Plot the attention maps
    num_layers = len(attention_maps)
    fig, axs = plt.subplots(1, num_layers, figsize=(16, 4))

    for i, attention_map in enumerate(attention_maps):
        axs[i].imshow(attention_map.squeeze(), cmap="viridis", aspect="auto")
        axs[i].set_title(f"Layer {i + 1} Attention")
        axs[i].axis("off")

    plt.show()
    return caption, output, attention_weights

# load displayer
device = 'cpu'
from PIL import Image
from IPython.display import display
from IPython.display import display
image_path = r'C:\Users\Asus\Desktop\Images\Caroline.jpg'

display(load_image(image_path))
caption = get_caption(model, image_processor, tokenizer, image_path)

print(len(caption), caption, 'end')