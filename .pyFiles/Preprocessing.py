from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision import transforms as TT
import torch
from Functions import FlickrDataset, CaptionTransform, collate_fn
from torch.utils.data import DataLoader


caption_transform = CaptionTransform()

with open('/content/drive/MyDrive/all_data/Flicker8k_Dataset/text/Flickr_8k.trainImages.txt') as f:
  lines = f.readlines()
  train_imgs = []
  for line in lines:
    name = line.split('#')[0].split('\n')[0]
    if name not in train_imgs:
      train_imgs.append(name)
with open('/content/drive/MyDrive/all_data/Flicker8k_Dataset/text/Flickr_8k.devImages.txt') as f:
  lines = f.readlines()
  valid_imgs = []
  for line in lines:
    name = line.split('#')[0].split('\n')[0]
    if name not in valid_imgs:
      valid_imgs.append(name)
with open('/content/drive/MyDrive/all_data/Flicker8k_Dataset/text/Flickr_8k.testImages.txt') as f:
  lines = f.readlines()
  test_imgs = []
  for line in lines:
    name = line.split('#')[0].split('\n')[0]
    if name not in test_imgs:
      test_imgs.append(name)

caps = []
with open('/content/drive/MyDrive/all_data/Flicker8k_Dataset/text/Flickr8k.token.txt') as f:
    lines = f.readlines()
for line in lines:
  caps.append(line.split('#')[1].replace('\t', '').replace('\n', '')[1:])

tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, caps), specials=['<pad>', '<unk>', '<eos>', '<sos>'])
vocab.set_default_index(vocab['<unk>'])
torch.save(vocab, '/content/drive/MyDrive/all_data/vocab.pt')

transform_train = TT.Compose([TT.Resize((256, 256)),
                              TT.CenterCrop(224),
                              TT.ToTensor(),
                              TT.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])

transform_test = TT.Compose([TT.Resize((224, 224)),
                            TT.ToTensor(),
                            TT.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

train = FlickrDataset('train', caption_transform, transform_train)
valid = FlickrDataset('valid', caption_transform, transform_test)
test = FlickrDataset('test', caption_transform, transform_test)


train_loader = DataLoader(train, 32, True, collate_fn=collate_fn)
valid_loader = DataLoader(valid, 64, False, collate_fn=collate_fn)
test_loader = DataLoader(test, 64, False, collate_fn=collate_fn)
