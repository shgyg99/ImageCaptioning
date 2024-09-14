import matplotlib.pyplot as plt
import torch

x, y = next(iter(test_loader))
idx = torch.randint(0, x.shape[0], (1,))
text = generate(x[[idx], ...], model, vocab, max_seq_length=20, device=device)
plt.imshow(x[[idx], ...].squeeze(0).permute(1, 2, 0))
print(text)