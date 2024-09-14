import streamlit as st
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as TT
from torchvision.transforms import functional as TF
from torchvision.models import resnet50, ResNet50_Weights
from torchtext.data.utils import get_tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = '.ptFiles\model.pt'
vocab = torch.load('./.ptFiles/vocab.pt')
tokenizer = get_tokenizer('basic_english')
embed_size=256
hidden_size=512
num_layers = 2
dropout_embd = 0.5
dropout_rnn = 0.5

class Encoder(nn.Module):
  def __init__(self, embed_size):
    super().__init__()
    self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    self.resnet.requires_grad_(False)
    feature_size = self.resnet.fc.in_features

    self.resnet.fc = nn.Identity()
    self.fc = nn.Linear(feature_size, embed_size)
    self.bn = nn.BatchNorm1d(embed_size)

  def forward(self, x):
    self.resnet.eval()
    with torch.no_grad():
      features = self.resnet(x)
    y = self.bn(self.fc(features))
    return y

class Decoder(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
    super(Decoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=vocab['<pad>'])
    self.dropout_embd = nn.Dropout(dropout_embd)

    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout_rnn, batch_first=True)

    self.linear = nn.Linear(hidden_size, vocab_size)

    self.max_seq_length = max_seq_length

  def init_weights(self):
    self.embedding.weight.data.uniform_(-0.1, 0.1)
    self.linear.bias.data.fill_(0)
    self.linear.weight.data.uniform_(-0.1, 0.1)

  def forward(self, features, captions):
    embeddings = self.dropout_embd(self.embedding(captions[:, :-1]))
    inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
    outputs, _ = self.lstm(inputs)
    outputs = self.linear(outputs)
    return outputs

  def generate(self, features, captions):
    if len(captions)!=0:
      embeddings = self.dropout_embd(self.embedding(captions))
      inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
    else:
      inputs = features.unsqueeze(1)
    outputs, _ = self.lstm(inputs)
    outputs = self.linear(outputs)
    return outputs

class ImageCaptioning(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
    super(ImageCaptioning, self).__init__()
    self.encoder = Encoder(embed_size)
    self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length)

  def forward(self, images, captions):
    features = self.encoder(images)
    output = self.decoder(features, captions)
    return output

  def generate(self, images, captions):
    features = self.encoder(images)
    output = self.decoder.generate(features, captions)
    return output
  

def generate(image, transform, model_path, vocab, max_seq_length, device):
    img_array = np.frombuffer(image, dtype=np.uint8)
    image_ = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    image_transformed = transform(TF.to_pil_image(image_)).unsqueeze(0)
    image = image_transformed.to(device)
    src, indices = [], []

    caption = ''
    itos = vocab.get_itos()

    for i in range(max_seq_length):
        with torch.no_grad():
            predictions = model.generate(image, src)

        idx = predictions[:, -1, :].argmax(1)
        token = itos[idx]
        caption += token + ' '

        if idx == vocab['<eos>']:
            break

        indices.append(idx)
        src = torch.LongTensor([indices]).to(device)

    return caption

ImageCaptioning(embed_size, hidden_size, len(vocab), num_layers, dropout_embd, dropout_rnn, 20)

test_transform = TT.Compose([TT.Resize((224, 224)),
                TT.ToTensor(),
                TT.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
st.set_page_config(
    page_title="Image Captioning Web App",
    page_icon="🖼",

)

st.title('Age Estimation Web App')

st.markdown("""
Welcome to our Age Estimation Web App! This application utilizes the powerful ResNet model, built on PyTorch,
 to accurately estimate the age of individuals from their photographs.
             Upload an image and let our state-of-the-art deep learning model predict the age in just a few seconds. 
            Experience the power of AI in age estimation!
* **Python libraries:** pytorch, pandas, streamlit, numpy
* **Data source:** [github-repository](https://github.com/shgyg99/Age-Estimation).
""")

st.write('---')



up = st.file_uploader('upload', type=['png', 'jpg'])
if up is not None:

    a = st.image(up)
    b = st.warning('just a sec')
    out = generate(up.read(), test_transform, model, vocab, 20, device)[5:-6]
    st.balloons()
    time.sleep(1)
    b.write(out)


    


st.write('------')
st.markdown('[send an email to me](mailto:shgyg99@gmail.com)',  unsafe_allow_html=True)