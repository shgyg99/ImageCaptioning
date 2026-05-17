import mlflow
import mlflow.pytorch
import os
from src.data_ingestion import DataIngestion
from src.custom_dataset import AutoTokenizer, CaptionTransform, FlickrDataset, transform_train, transform_test, collate_fn

import torch
from torch.utils.data import DataLoader
from config.data_ingestion_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
from torch.quantization import quantize_dynamic
from transformers import AutoTokenizer
from src.base_model import ImageCaptioning
import torch
from torch import nn
from torch import optim
import os
import matplotlib.pyplot as plt

from config.model_config import model, special_tokens_dict, train, dataloaders

from utils.common_functions import set_seed

from src.train import train_one_epoch, evaluate

logger = get_logger('main_pipeline')

if __name__=="__main__":

    mlflow.set_experiment("img-captioning")
    with mlflow.start_run(run_name="Training Pipeline"):

        mlflow.log_param("dataset_name", DATASET_NAME)
        mlflow.log_param("target_dir", TARGET_DIR)
        mlflow.log_param("train_batch_size", train['batch_size']/2)
        mlflow.log_param("valid_batch_size", train['batch_size'])
        mlflow.log_param("test_batch_size", train['batch_size'])
        mlflow.log_param("device", train['device'])
        mlflow.log_param("learning_rate", train['lr'])
        mlflow.log_param("num_epochs", train['num_epoch'])
        mlflow.log_param("seed", model['seed'])
        mlflow.log_param("embed_size", model['embed_size'])
        mlflow.log_param("hidden_size", model['hidden_size'])
        mlflow.log_param("num_layers", model['num_layers'])
        mlflow.log_param("dropout_embd", model['dropout_embd'])
        mlflow.log_param("dropout_rnn", model['dropout_rnn'])
        mlflow.log_param("seq_len", model['seq_len'])

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

            mlflow.log_param("train_dataset_size", len(train_dataset))
            mlflow.log_param("valid_dataset_size", len(valid_dataset))
            mlflow.log_param("test_dataset_size", len(test_dataset))

            logger.info("Datasets created successfully.")

        except Exception as e:
            logger.error(f"Error in creating datasets: {e}")
            raise CustomException(e)

        try:
            train_loader = DataLoader(train_dataset, int(train['batch_size']/2), True, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, train['batch_size'], False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, train['batch_size'], False, collate_fn=collate_fn)
            
            mlflow.log_param("train_steps", len(train_loader))
            mlflow.log_param("valid_steps", len(valid_loader))
            mlflow.log_param("test_steps", len(test_loader))

            logger.info("DataLoaders created successfully.")

        except Exception as e:
            logger.error(f"Error in creating DataLoaders: {e}")
            raise CustomException(e)

        if not os.path.exists('artifacts/dataloaders'):
            os.makedirs('artifacts/dataloaders')

        torch.save(train_loader, 'artifacts/dataloaders/train.pt')
        torch.save(valid_loader, 'artifacts/dataloaders/valid.pt')
        torch.save(test_loader, 'artifacts/dataloaders/test.pt')

        mlflow.log_artifact('artifacts/dataloaders/train.pt')
        mlflow.log_artifact('artifacts/dataloaders/valid.pt')
        mlflow.log_artifact('artifacts/dataloaders/test.pt')

        logger.info("DataLoaders saved successfully at 'artifacts/dataloader' path.")

        # -------------TRAINING THE MODEL---------------
        try:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            
            tokenizer.add_special_tokens(special_tokens_dict)
            
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

            mlflow.log_param("vocab_size", vocab_size)
            mlflow.log_param("pad_token_id", pad_token_id)

            logger.info("Model created successfully.")
        except Exception as e:
            logger.error(f"Error in creating model: {e}")
            raise CustomException(e)
        
        if not os.path.exists(train['model_path']):
            os.makedirs(train['model_path'])

        try:
            train_loader = torch.load(dataloaders['train'], weights_only=False)
            valid_loader = torch.load(dataloaders['valid'], weights_only=False)
            logger.info("DataLoaders loaded successfully.")
        except Exception as e:
            logger.error(f"Error in loading DataLoaders: {e}")
            raise CustomException(e)

        set_seed(model['seed'])
        inputs, targets = next(iter(train_loader))
        inputs = inputs.to(train['device'])
        targets = targets.to(train['device'])

        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        optimizer = optim.AdamW(img_model.parameters(), lr=train['lr'])
        metric = None

        loss_train_hist = []
        loss_valid_hist = []

        best_loss_valid = torch.inf
        epoch_counter = 0

        num_epochs = train['num_epoch']

        for epoch in range(num_epochs):
            # Train
            img_model, loss_train, _ = train_one_epoch(img_model,
                                                    train_loader,
                                                    loss_fn,
                                                    optimizer,
                                                    metric,
                                                    epoch)
            # Validation
            loss_valid, _ = evaluate(img_model,
                                    valid_loader,
                                    loss_fn,
                                    metric)

            loss_train_hist.append(loss_train)
            loss_valid_hist.append(loss_valid)

            mlflow.log_metric("train_loss", loss_train, step=epoch)
            mlflow.log_metric("valid_loss", loss_valid, step=epoch)

            if loss_valid < best_loss_valid:
                quantized_model = quantize_dynamic(
                                    img_model,
                                    {torch.nn.Linear},
                                    dtype=torch.qint8
                                )
                model_path = os.path.join(train['model_path'], f'final_model.pt')
                torch.save(quantized_model, model_path)
                mlflow.log_artifact(model_path)
                best_loss_valid = loss_valid
                logger.info(f"Model saved at epoch {epoch} with validation loss: {loss_valid:.4}")

            epoch_counter += 1
        
        plt.figure(figsize=(8, 6))

        plt.plot(range(epoch_counter), loss_train_hist, 'r-', label='Train')
        plt.plot(range(epoch_counter), loss_valid_hist, 'b-', label='Validation')

        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(train['model_path'], 'loss_plot.png'))
        mlflow.log_artifact(os.path.join(train['model_path'], 'loss_plot.png'))
