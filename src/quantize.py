import torch.quantization as quant
import torch
from src.base_model import *

model = torch.load('artifacts/checkpoints/final_model.pt', weights_only=False, map_location=torch.device('cpu'))
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {
        torch.nn.Linear,   
        torch.nn.LSTM,      
    },
    dtype=torch.qint8
)

torch.save(quantized_model, 'artifacts/checkpoints/quant_model.pt')