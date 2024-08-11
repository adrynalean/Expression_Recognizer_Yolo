# MoodWave

![MoodWave](https://cdn.dribbble.com/users/38593/screenshots/2993662/emoji-all-dribbb.gif)

This project is a deep learning-based expression classifier using PyTorch. The classifier is designed to recognize facial expressions from images and is trained on a dataset that contains several classes of emotions.

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Setup and Installation

### Dependencies

The following dependencies are required to run the project:

- Python 3.x
- PyTorch
- Torchvision
- OpenCV
- Matplotlib

You can install these dependencies using pip:

```bash
pip install torch torchvision torchaudio opencv-python matplotlib
```

### Device Setup

The project is device agnostic and can run on both CPU and GPU. The device is automatically detected by the following code snippet:

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```

## Data Preparation

The dataset used for training the model contains images categorized by emotions (e.g., angry, happy, sad). These images are stored in zip files, which need to be extracted and organized into appropriate directories.

### Steps:

1. **Extract the Dataset**: The dataset is extracted from zip files and organized into separate folders for each emotion.
2. **Data Cleaning**: Any dodgy or corrupted images are removed from the dataset using OpenCV and `imghdr`.
3. **Data Transformation**: Images are resized to a uniform size and transformed into tensors for training.

### Example Code:

```python
import zipfile
import os
import shutil
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Extracting zip files and organizing datasets
# (See the notebook for detailed code)
```

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

- **Conv Layer 1**: 3 input channels, 16 output channels, 3x3 kernel
- **Conv Layer 2**: 16 input channels, 32 output channels, 3x3 kernel
- **Conv Layer 3**: 32 input channels, 64 output channels, 3x3 kernel
- **Pooling Layer**: Max Pooling with 2x2 kernel
- **Fully Connected Layer 1**: 128 neurons
- **Output Layer**: Number of classes neurons (equal to the number of emotion categories)

### Example Code:

```python
import torch.nn as nn
import torch.nn.functional as F

class AICModel(nn.Module):
    def __init__(self):
        super(AICModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, len(dataset.classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Training the Model

The model is trained using Cross Entropy Loss and Adam optimizer. The training process includes 5 epochs and involves shuffling the data for each epoch.

### Example Code:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop (See the notebook for detailed code)
```

## Results

The model's performance is evaluated based on accuracy, and example images from the test dataset are visualized along with their predicted labels.

## Usage

To use this project, follow these steps:

1. Clone the repository.
2. Install the dependencies.
3. Run the notebook to train the model.
4. Use the trained model to classify new images.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

