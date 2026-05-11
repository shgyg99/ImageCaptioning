import os
from src.data_ingestion import DataIngestion
from src.custom_dataset import AutoTokenizer, CaptionTransform, FlickrDataset, transform_train, transform_test, collate_fn

import torch
from torch.utils.data import DataLoader
from config.data_ingestion_config import *
from config.model_config import train
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger('processing pipeline')


if __name__=="__main__":
    # -------------DATA INGESTION---------------
    if not os.path.exists(os.path.join(TARGET_DIR, 'raw', 'captions.txt')):
        data_ingestion = DataIngestion(DATASET_NAME,TARGET_DIR)
        data_ingestion.run()

    # -------------CREATING DATA LOADER FILES---------------
    try:
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

        logger.info("Datasets created successfully.")

    except Exception as e:
        logger.error(f"Error in creating datasets: {e}")
        raise CustomException(e)

    try:
        train_loader = DataLoader(train_dataset, train['batch_size']/2, True, collate_fn=collate_fn ,num_workers=2, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, train['batch_size'], False, collate_fn=collate_fn, num_workers=2)
        test_loader = DataLoader(test_dataset, train['batch_size'], False, collate_fn=collate_fn)
        
        logger.info("DataLoaders created successfully.")

    except Exception as e:
        logger.error(f"Error in creating DataLoaders: {e}")
        raise CustomException(e)
    
    if not os.path.exists('artifacts/dataloaders'):
        os.makedirs('artifacts/dataloaders')

    torch.save(train_loader, 'artifacts/dataloaders/train.pt')
    torch.save(valid_loader, 'artifacts/dataloaders/valid.pt')
    torch.save(test_loader, 'artifacts/dataloaders/test.pt')
    
    logger.info("DataLoaders saved successfully at 'artifacts/dataloader' path.")


