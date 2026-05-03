import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as TT

from PIL import Image

from transformers import AutoTokenizer

from src.logger import get_logger
from src.custom_exception import CustomException


logger = get_logger('custom_dataset')


# -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------CUSTOM DATASET-------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
class FlickrDataset(Dataset):
  def __init__(self, phase, image_transform=None, target_transform=None, img_root='artifacts/raw/Images', caption_path='artifacts/raw/captions.txt'):
    self.root = img_root
    self.cap_path = caption_path
    self.image_transform = image_transform
    self.target_transform = target_transform

    try:
      with open(self.cap_path) as f:
        lines = f.readlines()
        images = []
        for line in lines:
          name = line.split('#')[0].split('\n')[0]
          if name not in images:
            images.append(name)

      self.amounts = {'train': images[1:28310], 'valid': images[28310:36400], 'test': images[36400:]}
      self.images = self.amounts[phase]

      self.text, self.image = [], []
      for case in self.images:
        try:
          img, cap = case.split(',')
        except:
          img, cap = case.split(',')[0], ','.join(case.split(',')[1:])

        self.text.append(cap)
        self.image.append(img)
      
      logger.info(f'Initialized {phase} dataset with {len(self.text)} samples.')

    except Exception as e:
      logger.error(f'phase {phase} Error initializing dataset: {e}')
      raise CustomException(f'Error initializing dataset: {e}')
       

  def __getitem__(self, index):
    image_path = os.path.join(self.root, self.image[index])
    img = Image.open(image_path).convert('RGB')
    caption = self.text[index]

    if self.image_transform:
      try:
        img = self.image_transform(img)
        logger.info(f'Image transformed successfully for index {index}.')
      except Exception as e:
         logger.error(f'Error transforming image at index {index}: {e}')
         raise CustomException(f'Error transforming image at index {index}: {e}')

    if self.target_transform:
      try:
        caption = self.target_transform(caption)
        logger.info(f'Caption transformed successfully for index {index}.')
      except Exception as e:
         logger.error(f'Error transforming caption at index {index}: {e}')
         raise CustomException(f'Error transforming caption at index {index}: {e}')

    return img, caption
  
  def __len__(self):
      return len(self.text)


# -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------CAPTION TRANSFORMER-------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
class CaptionTransform:
    def __init__(self, tokenizer, max_length=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, caption):
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoding['input_ids'].squeeze(0)
    
    def decode(self, indices):
        return self.tokenizer.decode(indices, skip_special_tokens=True)
    
    def __repr__(self):
        return f"""CaptionTransform([
            tokenizer='{self.tokenizer.name_or_path}',
            vocab_size={self.tokenizer.vocab_size},
            max_length={self.max_length}
        ])"""
    


# -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------IMAGE TRANSFORMER-------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
transform_train = TT.Compose([TT.Resize((256, 256)),
                              TT.CenterCrop(224),
                              TT.ToTensor(),
                              TT.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])

transform_test = TT.Compose([TT.Resize((224, 224)),
                            TT.ToTensor(),
                            TT.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])



# -------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------DATA LOADER-------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
def collate_fn(data):
    images = []
    captions = []
    
    for img, cap in data:
        images.append(img)
        captions.append(cap)
    
    batched_images = torch.stack(images)
    batched_captions = torch.stack(captions)
    
    return batched_images, batched_captions



if __name__=='__main__':
  tokenizer = AutoTokenizer.from_pretrained('gpt2')

  special_tokens = {
      'bos_token': '<sos>',
      'eos_token': '<eos>',
      'pad_token': '<pad>',
      'unk_token': '<unk>'
  }
  tokenizer.add_special_tokens(special_tokens)
  caption_transform = CaptionTransform(tokenizer, max_length=50)
  train_dataset = FlickrDataset(phase='train', image_transform=transform_train, target_transform=caption_transform)
  valid_dataset = FlickrDataset(phase='valid', image_transform=transform_test, target_transform=caption_transform)
  test_dataset = FlickrDataset(phase='test', image_transform=transform_test, target_transform=caption_transform)


  train_loader = DataLoader(train_dataset, 16, True, collate_fn=collate_fn)
  valid_loader = DataLoader(valid_dataset, 32, False, collate_fn=collate_fn)
  test_loader = DataLoader(test_dataset, 32, False, collate_fn=collate_fn)
  
  if not os.path.exists('artifacts/dataloaders'):
     os.makedirs('artifacts/dataloaders')
     logger.info(f'artifacts/dataloaders created successfully...')
  try:
    torch.save(train_loader, 'artifacts/dataloaders/train.pt')
    torch.save(valid_loader, 'artifacts/dataloaders/valid.pt')
    torch.save(test_loader, 'artifacts/dataloaders/test.pt')

    logger.info('Data Loaders saved in artifacts/dataloaders')

  except Exception as e:
     logger.error('Error while saving data loaders...{e}')
     raise CustomException(f'Error while saving dataloaders {e}')

