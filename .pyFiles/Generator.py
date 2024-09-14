import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Arguments import device
from Functions import generate

model = torch.load('.ptFiles/model.pt', map_location=device)
model.eval()
test_loader = torch.load('.ptFiles/test.pt')
vocab = torch.load('.ptFiles/vocab.pt')
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

x, y = next(iter(test_loader))
idx = torch.randint(0, x.shape[0], (1,))
text = generate(x[[idx], ...], model, vocab, max_seq_length=20, device=device)
plt.imshow(x[[idx], ...].squeeze(0).permute(1, 2, 0))
print(text)