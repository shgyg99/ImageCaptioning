import torch
from transformers import AutoTokenizer
from functools import lru_cache
from PIL import Image
from src.custom_dataset import transform_test
from src.base_model import ImageCaptioning, Encoder, Decoder
from utils.config_manager import ConfigManager
from utils.google_drive_downloader import GoogleDriveDownloader

config_manager = ConfigManager.from_yaml()
model_config = config_manager.get("model")
train_config = config_manager.get("train")
special_tokens_dict = config_manager.get("special_tokens_dict")

device = train_config.get("device", "cpu")

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens(special_tokens_dict)
vocab_size = tokenizer.vocab_size + len(special_tokens_dict)
pad_token_id = tokenizer.pad_token_id
eos_token_id = 764

def load_model(model_path='artifacts/img_captioning.pt'):
    model = ImageCaptioning(
        embed_size=model_config.get('embed_size'),
        hidden_size=model_config.get('hidden_size'),
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        num_layers=model_config.get('num_layers'),
        dropout_embd=model_config.get('dropout_embd'),
        dropout_rnn=model_config.get('dropout_rnn'),
        max_seq_length=model_config.get('seq_len')
    ).to(device)



    try:
        model_file = GoogleDriveDownloader.get_model_path(model_path)

        state_dict = torch.load(model_file, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Model loaded successfully from {model_file}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("⚠️ Running without model - predictions will not work")

    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

img_model = load_model()

@lru_cache(maxsize=10000)
def fast_decode(tokenizer, token_ids_tuple):
    return tokenizer.decode(list(token_ids_tuple), skip_special_tokens=True)

@torch.no_grad()
def caption_generation(image):
    image = image.to(device)
    indices = []
    
    for i in range(model_config.get("seq_len")):
        src = torch.LongTensor([indices]).to(device) if indices else None
        predictions = img_model.generate(image, src)
        idx = predictions[:, -1, :].argmax(1)
        
        if idx.item() == eos_token_id:
            break
        indices.append(idx.item())
    
    caption = fast_decode(tokenizer, tuple(indices))
    return caption