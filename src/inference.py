import torch
from config.model_config import model, special_tokens_dict, train
from transformers import AutoTokenizer
from functools import lru_cache
from PIL import Image
from src.custom_dataset import transform_test
from src.base_model import ImageCaptioning, Encoder, Decoder

@lru_cache(maxsize=10000)
def fast_decode(tokenizer, token_ids_tuple):
    return tokenizer.decode(list(token_ids_tuple), skip_special_tokens=True)

@torch.no_grad() 
def caption_generation(image):

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens(special_tokens_dict)
    eos_token_id = 764

    vocab_size = tokenizer.vocab_size + len(special_tokens_dict)  # Adding 4 for the special tokens
    pad_token_id = tokenizer.pad_token_id 
    
    img_model = ImageCaptioning(
        embed_size=model['embed_size'],
        hidden_size=model['hidden_size'],
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        num_layers=model['num_layers'],
        dropout_embd=model['dropout_embd'],
        dropout_rnn=model['dropout_rnn'],
        max_seq_length=model['seq_len']
    ).to(train['device'])

    img_model.load_state_dict(torch.load('artifacts/checkpoints/final_model.pt', map_location=train['device']))
    img_model.eval()

    

    image = image.to(train['device'])
    indices = []
    
    for i in range(model['seq_len']):
        with torch.no_grad():
            src = torch.LongTensor([indices]).to(train['device']) if indices else None
            predictions = img_model.generate(image, src)
            
            idx = predictions[:, -1, :].argmax(1)
            
            if idx.item() == eos_token_id:
                break
                
            indices.append(idx.item())
    
    caption = fast_decode(tokenizer, tuple(indices))
    return caption
    

if __name__ == "__main__":
    # Example usage
    image_path = "artifacts/raw/Images/44856031_0d82c2c7d1.jpg"
    image = Image.open(image_path).convert('RGB')
    transformed_img = transform_test(image)
    transformed_img = torch.stack([transformed_img])
    caption = caption_generation(transformed_img)
    print(f"Generated Caption: {caption}")