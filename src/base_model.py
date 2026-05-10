import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoTokenizer
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger('base_model')

class Encoder(nn.Module):
  def __init__(self, embed_size):
    super().__init__()
    try:
      self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
      logger.info('ResNet50 model loaded successfully.')
    except Exception as e:
      logger.error(f'Error loading ResNet50 model: {e}')
      raise CustomException(f'Error loading ResNet50 model: {e}')

    self.resnet.requires_grad_(False)
    feature_size = self.resnet.fc.in_features

    self.resnet.fc = nn.Identity()
    self.fc = nn.Linear(feature_size, embed_size)
    self.bn = nn.BatchNorm1d(embed_size)

  def forward(self, x):
    self.resnet.eval()
    with torch.no_grad():
      features = self.resnet(x)
    y = self.bn(self.fc(features))
    return y
  

class Decoder(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, pad_index, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
    super(Decoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_index)
    self.dropout_embd = nn.Dropout(dropout_embd)

    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout_rnn, batch_first=True)

    self.linear = nn.Linear(hidden_size, vocab_size)

    self.max_seq_length = max_seq_length

  def init_weights(self):
    self.embedding.weight.data.uniform_(-0.1, 0.1)
    self.linear.bias.data.fill_(0)
    self.linear.weight.data.uniform_(-0.1, 0.1)

  def forward(self, features, captions):
    embeddings = self.dropout_embd(self.embedding(captions[:, :-1]))
    inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
    outputs, _ = self.lstm(inputs)
    outputs = self.linear(outputs)
    return outputs

  def generate(self, features, captions):

    if captions is not None:
      embeddings = self.dropout_embd(self.embedding(captions))
      inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
    else:
      inputs = features.unsqueeze(1)
    outputs, _ = self.lstm(inputs)
    outputs = self.linear(outputs)
    return outputs
  

class ImageCaptioning(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, pad_token_id, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
    super(ImageCaptioning, self).__init__()
    try:
      self.encoder = Encoder(embed_size)
      logger.info('Encoder initialized successfully.')
      self.decoder = Decoder(embed_size, hidden_size, vocab_size, pad_token_id, num_layers, dropout_embd, dropout_rnn, max_seq_length)
      logger.info('Decoder initialized successfully.')
    except Exception as e:
      logger.error(f'Error initializing ImageCaptioning model: {e}')
      raise CustomException(f'Error initializing ImageCaptioning model: {e}') 

  def forward(self, images, captions):
    features = self.encoder(images)
    output = self.decoder(features, captions)
    return output

  def generate(self, images, captions):
    features = self.encoder(images)
    output = self.decoder.generate(features, captions)
    return output
  


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    special_tokens_dict = {
        'bos_token': '<sos>',
        'eos_token': '<eos>',
        'pad_token': '<pad>',
        'unk_token': '<unk>'
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    vocab_size = tokenizer.vocab_size + 4  # Adding 4 for the special tokens
    pad_token_id = tokenizer.pad_token_id 
    
    print(f"Vocab size: {vocab_size}")
    print(f"Pad token ID: {pad_token_id}")
    
    model = ImageCaptioning(
        embed_size=300,
        hidden_size=500,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        num_layers=2,
        dropout_embd=0.5,
        dropout_rnn=0.5,
        max_seq_length=20
    )
    test_loader = torch.load('artifacts/dataloaders/test.pt', weights_only=False)
    x_temp, y_temp = next(iter(test_loader))

    out = model(x_temp, y_temp)
    print(out.shape)
    