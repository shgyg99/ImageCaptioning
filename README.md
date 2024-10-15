
---

# Image Captioning with PyTorch

This repository contains the implementation of an image captioning model using PyTorch. The model is trained on the Flickr8k dataset but **the dataset files are not included** in this repository. If you wish to use this code, please download the dataset yourself from [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) and store it locally. You will need to set the correct dataset path in the code.

## Requirements
- PyTorch
- Streamlit
- Other dependencies can be found in `requirements.txt`.

## Dataset Setup
1. Download the Flickr8k dataset.
2. Store the dataset in your preferred local directory.
3. Update the dataset path in the corresponding `.py` files.

## Web App
This repository includes a **Streamlit web app** that generates captions for images using a pre-trained model.
### How to run with *Docker* :
you have to go to streamlit folder with:
```cmd
cd streamlit
```
then build docker image with :
```cmd
docker build -t image-captioning .
```


Check out the live demo of the app on Hugging Face Spaces by clicking the image below:

[![Hugging Face Space](https://raw.githubusercontent.com/shgyg99/ImageCaptioning/master/screenshot20240916085613.png)](https://shgyg99-imagecaptioning.hf.space)

## Model Training
The training code is included in the `.pyFiles` folder. You can customize the training settings such as batch size, learning rate, and the number of epochs.

To train the model:
```bash
python .pyFiles/MainTrainLoop.py
```

Make sure the dataset is correctly set up before starting the training.


## Note
- The `.ptFiles` folder contains pre-trained model weights and train_valid_testloder and vocabulary .
- Modify the paths in the code to match your local setup.

---

Feel free to send an email to me : [shgyg99@gmail.com](mailto:shgyg99@gmail.com)
