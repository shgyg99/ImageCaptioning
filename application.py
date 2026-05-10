from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
from src.custom_dataset import transform_test
from src.inference import caption_generation


app = Flask(__name__, template_folder="templates")

MODEL_PATH = "artifacts/checkpoints/loss_4.21.pt"

model = torch.load(MODEL_PATH, weights_only=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "There is no chosen file yet", 400
    try:
        file = request.files['image']
        
        image_bytes = file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transformed_img = transform_test(image)
        transformed_img = torch.stack([transformed_img])
        caption = caption_generation(transformed_img)

        return render_template('index.html', prediction_text = caption)
    
    except Exception as e:
        return jsonify({'error' : str(e)})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
