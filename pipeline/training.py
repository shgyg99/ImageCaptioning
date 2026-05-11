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

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger('training pipeline')


if __name__=="__main__":

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



        if loss_valid < best_loss_valid:
            quantized_model = quantize_dynamic(
                                img_model,
                                {torch.nn.Linear},
                                dtype=torch.qint8
                            )
            torch.save(quantized_model, os.path.join(train['model_path'], f'final_model.pt'))
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
    plt.show()




    