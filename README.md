# Image Captioning

An end-to-end deep learning project that generates descriptive captions for images using encoder-decoder architecture with ResNet50 and transformer-based models.

## Overview

This project implements an image captioning system that automatically generates natural language descriptions for images. It uses a CNN encoder (ResNet50) to extract visual features and a transformer decoder to generate captions.

## Project Structure

```
img-captioning/
├── src/                          # Core source code
│   ├── base_model.py             # Encoder and decoder model architectures
│   ├── custom_dataset.py         # PyTorch Dataset class
│   ├── data_ingestion.py         # Download and process raw data from Kaggle
│   ├── custom_exception.py       # Custom exception handling
│   ├── logger.py                 # Logging configuration
│   └── __init__.py
│
├── config/                       # Configuration files
│   ├── data_ingestion_config.py  # Data paths and parameters
│   ├── model_config.py           # Model hyperparameters
│   └── __init__.py
│
├── pipeline/                     # Data processing pipelines
│   └── processing.py             # Data preprocessing and augmentation
│
├── utils/                        # Utility functions
│   ├── common_functions.py       # Helper functions
│   └── __init__.py
│
├── artifacts/                    # Generated outputs
│   ├── raw/                      # Raw dataset (images & captions)
│   │   ├── Images/               # Image files
│   │   └── captions.txt          # Captions text file
│   └── dataloaders/              # Serialized PyTorch data loaders
│       ├── train.pt              # Training data loader
│       ├── valid.pt              # Validation data loader
│       └── test.pt               # Test data loader
│
├── logs/                         # Application logs
├── NoteBook.ipynb                # Jupyter notebook for experimentation
├── setup.py                      # Package setup configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Folder Descriptions

| Folder | Purpose |
|--------|---------|
| **src/** | Main source code containing models, data processing, and logging utilities |
| **config/** | Configuration parameters for data ingestion and model training |
| **pipeline/** | Data processing and preprocessing logic |
| **utils/** | Reusable utility functions across the project |
| **artifacts/** | Raw datasets and pre-processed data loaders |
| **logs/** | Application execution logs |

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd img-captioning
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -e .
```

For CPU-only PyTorch installation:
```bash
pip install -e . --index-url https://download.pytorch.org/whl/cpu
```

## Usage

Explore the `NoteBook.ipynb` for end-to-end workflow including:
- Data ingestion from Kaggle
- Model building and training
- Caption generation

## Dependencies

- **torch** 2.6.0 - Deep learning framework
- **torchvision** 0.21.0 - Computer vision utilities
- **transformers** - Pre-trained models and tokenizers
- **kagglehub** - Kaggle dataset download
- **Pillow** - Image processing
- **numpy** - Numerical computing
