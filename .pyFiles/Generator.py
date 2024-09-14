import matplotlib.pyplot as plt
import torch

model = torch.load('/content/drive/MyDrive/all_data/model.pt', map_location=device)
model.eval()
test_loader = torch.load('/content/drive/MyDrive/all_data/test.pt')
vocab = torch.load('/content/drive/MyDrive/all_data/vocab.pt')
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

x, y = next(iter(test_loader))
idx = torch.randint(0, x.shape[0], (1,))
text = generate(x[[idx], ...], model, vocab, max_seq_length=20, device=device)
plt.imshow(x[[idx], ...].squeeze(0).permute(1, 2, 0))
print(text)