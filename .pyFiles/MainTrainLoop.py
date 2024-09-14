import torch
from torch import nn
from torch import optim
from torchtext.data.utils import get_tokenizer
from Functions import set_seed, ImageCaptioning, train_one_epoch, evaluate
from Arguments import seed, embed_size, hidden_size, num_epoch
from Arguments import num_layers, dropout_embd, dropout_rnn, device, lr
from Preprocessing import train_loader, valid_loader


vocab = torch.load('./.ptFiles/vocab.pt')
tokenizer = get_tokenizer('basic_english')
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
metric = None

torch.cuda.empty_cache()
set_seed(seed)
model = ImageCaptioning(embed_size, hidden_size,
                        len(vocab), num_layers, 
                        dropout_embd, dropout_rnn).to(device)

"""🔰 Define optimizer and Set learning rate and weight decay."""

set_seed(seed)
# wd = 0
# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
optimizer = optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
metric = None

loss_train_hist = []
loss_valid_hist = []


best_loss_valid = torch.inf
epoch_counter = 0

num_epochs = num_epoch

for epoch in range(num_epochs):
  # Train
  model, loss_train, _ = train_one_epoch(model,
                                        train_loader,
                                        loss_fn,
                                        optimizer,
                                        metric,
                                        epoch)
  # Validation
  loss_valid, _ = evaluate(model,
                          valid_loader,
                          loss_fn,
                          metric)

  loss_train_hist.append(loss_train)
  loss_valid_hist.append(loss_valid)



  if loss_valid < best_loss_valid:
    torch.save(model, f'/content/drive/MyDrive/all_data/model.pt')
    best_loss_valid = loss_valid
    print('Model Saved!')

  print(f'Valid: Loss = {loss_valid:.4}')
  print()

  epoch_counter += 1