model = {
    'seed' : 8,
    'embed_size':256,
    'hidden_size':512,
    'num_layers' : 2,
    'dropout_embd' : 0.5,
    'dropout_rnn' : 0.5,
    'seq_len' : 20
}

train = {
    'batch_size':32,
    'lr' : 5e-3,
    'momentum':0.9,
    'num_epoch' : 4,
    'device': 'cpu',
    'model_path': 'artifacts/checkpoints'

}


dataloaders = {
    'train': 'artifacts/dataloaders/train.pt',
    'valid': 'artifacts/dataloaders/valid.pt',
    'test': 'artifacts/dataloaders/test.pt'
}

special_tokens_dict = {
        'bos_token': '<sos>',
        'eos_token': '<eos>',
        'pad_token': '<pad>',
        'unk_token': '<unk>'
    }