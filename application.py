from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import base64
from src.custom_dataset import transform_test
from src.inference import caption_generation
from src.base_model import ImageCaptioning, Encoder, Decoder
import os

app = Flask(__name__, template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    try:
        file = request.files['image']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        transformed_img = transform_test(image)
        transformed_img = torch.stack([transformed_img])
        caption = caption_generation(transformed_img)
        
        return jsonify({
            'caption': caption,
            'image': img_str
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)