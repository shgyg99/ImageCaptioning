import torch

seed = 8

embed_size=256
hidden_size=512
num_layers = 2
dropout_embd = 0.5
dropout_rnn = 0.5

batch_size=128
lr = 1e-2
momentum=0.9
num_epoch = 10

seq_len = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'