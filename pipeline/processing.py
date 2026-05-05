import os
from src.data_ingestion import DataIngestion
from src.custom_dataset import AutoTokenizer, CaptionTransform, FlickrDataset, transform_train, transform_test, collate_fn

import torch
from torch.utils.data import DataLoader
from config.data_ingestion_config import *





if __name__=="__main__":
    # -------------DATA INGESTION---------------
    if not os.path.exists(os.path.join(TARGET_DIR, 'raw', 'captions.txt')):
        data_ingestion = DataIngestion(DATASET_NAME,TARGET_DIR)
        data_ingestion.run()

    # -------------CREATING DATA LOADER FILES---------------
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

    torch.save(train_loader, 'artifacts/dataloaders/train.pt')
    torch.save(valid_loader, 'artifacts/dataloaders/valid.pt')
    torch.save(test_loader, 'artifacts/dataloaders/test.pt')

