import os
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from torchvision.models import resnet50, ResNet50_Weights
from torchtext.data.utils import get_tokenizer
import torch
from torch import nn
from torch.utils.data import Dataset
import tqdm
from PIL import Image
import numpy as np
from Preprocessing import train_imgs, valid_imgs, test_imgs
from Arguments import device

vocab = torch.load('.ptFiles/vocab.pt')
tokenizer = get_tokenizer('basic_english')
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def num_trainable_params(model):
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums

def set_seed(seed):

  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)

class FlickrDataset(Dataset):
  def __init__(self, phase, target_transform=None, image_transform=None, root='/content/drive/MyDrive/all_data/Flicker8k_Dataset/image/'):
    self.root = root
    self.target_transform = target_transform
    self.image_transform = image_transform
    if phase=='train':
      images = train_imgs
    elif phase=='valid':
      images = valid_imgs
    elif phase=='test':
      images = test_imgs
    else:
      raise print('your entered phase must be one of "train" or "valid" or "test"')

    self.images = images

    with open('/content/drive/MyDrive/all_data/Flicker8k_Dataset/text/Flickr8k.token.txt') as f:
      lines = f.readlines()
    self.text = []
    self.image = []
    for im_name in images:
      for line in lines:
        if line.split('#')[0]==im_name:
          self.text.append(line.split('#')[1].replace('\t', '').replace('\n', '')[1:])
          self.image.append(im_name)


  def __getitem__(self, index):
    image_path = os.path.join(self.root, self.image[index])
    img = Image.open(image_path).convert('RGB')
    caption = self.text[index]

    if self.image_transform:
      img = self.image_transform(img)

    if self.target_transform:
      caption = self.target_transform(caption)

    return img, caption


  def __len__(self):
    return len(self.text)

class CaptionTransform:
  def __init__(self, vocab=vocab, tokenizer=tokenizer):
    self.vocab = vocab
    self.tokenizer = tokenizer

  def __call__(self, caption):
    indices = self.vocab(self.tokenizer(caption))
    target = torch.LongTensor(self.vocab(['<sos>']) + indices + self.vocab(['<eos>']))
    return target

  def __repr__(self):
    return f"""CaptionTransform([
      _load_captions(),
      tokenizer('basic_english'),
      vocab(vocab_size={len(self.vocab)})
    ])"""

def collate_fn(data):
    tensors, targets = zip(*data)
    features = pad_sequence(targets, padding_value=vocab['<pad>'], batch_first=True)
    tensors = torch.stack(tensors)
    return tensors, features

class Encoder(nn.Module):
  def __init__(self, embed_size):
    super().__init__()
    self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
    super(Decoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=vocab['<pad>'])
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
    if len(captions)!=0:
      embeddings = self.dropout_embd(self.embedding(captions))
      inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
    else:
      inputs = features.unsqueeze(1)
    outputs, _ = self.lstm(inputs)
    outputs = self.linear(outputs)
    return outputs

class ImageCaptioning(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
    super(ImageCaptioning, self).__init__()
    self.encoder = Encoder(embed_size)
    self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length)

  def forward(self, images, captions):
    features = self.encoder(images)
    output = self.decoder(features, captions)
    return output

  def generate(self, images, captions):
    features = self.encoder(images)
    output = self.decoder.generate(features, captions)
    return output

def train_one_epoch(model, train_loader, loss_fn, optimizer, metric=None, epoch=None):
  model.train()
  loss_train = AverageMeter()
  if metric:
    metric.reset()

  with tqdm.tqdm(train_loader, unit='batch') as tepoch:
    for inputs, targets in tepoch:
      if epoch:
        tepoch.set_description(f'Epoch {epoch}')

      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs, targets)

      loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

      loss_train.update(loss.item(), n=len(targets))
      if metric:
        metric.update(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())


      tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item() if metric else None)
  return model, loss_train.avg, metric.compute().item() if metric else None


def evaluate(model, test_loader, loss_fn, metric=None):
  model.eval()
  loss_eval = AverageMeter()
  if metric:
    metric.reset()

  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs, targets)

      loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
      loss_eval.update(loss.item(), n=len(targets))

      if metric:
        metric.update(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

      return loss_eval.avg, metric.compute().item() if metric else None

def generate(image, model, vocab, max_seq_length, device):
  image = image.to(device)
  src, indices = [], []

  caption = ''
  itos = vocab.get_itos()

  for i in range(max_seq_length):
    with torch.no_grad():
      predictions = model.generate(image, src)

    idx = predictions[:, -1, :].argmax(1)
    token = itos[idx]
    caption += token + ' '

    if idx == vocab['<eos>']:
      break

    indices.append(idx)
    src = torch.LongTensor([indices]).to(device)

  return caption