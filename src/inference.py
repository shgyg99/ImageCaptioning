import torch
import os
from torch import nn
import matplotlib.pyplot as plt
from torch.quantization import quantize_dynamic
from config.model_config import model, special_tokens_dict, train, dataloaders
from transformers import AutoTokenizer
from functools import lru_cache
from PIL import Image
from src.custom_dataset import transform_test

@lru_cache(maxsize=10000)
def fast_decode(tokenizer, token_ids_tuple):
    return tokenizer.decode(list(token_ids_tuple), skip_special_tokens=True)

@torch.no_grad() 
def caption_generation(image):

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(special_tokens_dict)
    eos_token_id = tokenizer.eos_token_id

    quantized_model = torch.load('artifacts/checkpoints/quan.pt', map_location=train['device'], weights_only=False)
    quantized_model.eval()
    

    image = image.to(train['device'])
    indices = []
    
    for i in range(model['seq_len']):
        with torch.no_grad():
            src = torch.LongTensor([indices]).to(train['device']) if indices else None
            predictions = quantized_model.generate(image, src)
            
            idx = predictions[:, -1, :].argmax(1)
            
            if idx.item() == eos_token_id:
                break
                
            indices.append(idx.item())
    
    caption = fast_decode(tokenizer, tuple(indices))
    return caption
    

if __name__ == "__main__":
    # Example usage
    image_path = "artifacts/raw/Images/12830823_87d2654e31.jpg"
    image = Image.open(image_path).convert('RGB')
    transformed_img = transform_test(image)
    transformed_img = torch.stack([transformed_img])
    caption = caption_generation(transformed_img)
    print(f"Generated Caption: {caption}")