"""
Test trainer.
"""

# Imports ---------------------------------------------------------------------

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ColorJitter
from torchvision.transforms import RandomErasing
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

from firekit.preprocess import split_dataframe
from firekit.train import Trainer
from firekit.train.metrics import Accuracy
from firekit.train.metrics import Precision
from firekit.train.metrics import Recall
from firekit.train.metrics import F1
from firekit.vision import ImagePathDataset
from firekit.vision.transforms import SquarePad

# Constants -------------------------------------------------------------------

SHAPES_DIR = os.path.join("tests", "datasets", "shapes")
IMAGES_DIR = os.path.join(SHAPES_DIR, "images")
DATASET_CSV = os.path.join(SHAPES_DIR, "labels.csv")
BINARY_CNN_MODEL_PATH = os.path.join("tests", "models", "binary-cnn.pt")

# CNN -------------------------------------------------------------------------

class BinaryCNN(nn.Module):

    """
    A simple CNN classifier.
    """

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(3, 64, 7, padding="same")
        self.conv2 = nn.Conv2d(64, 128, 5, padding="same")
        self.conv3 = nn.Conv2d(128, 128, 3, padding="same")
        self.fc1 = nn.Linear(128 * 62 * 62, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SchemeTwoCNN(nn.Module):

    """
    A deep CNN with three sets of convolution layers separated by pooling. 
    Use learning rate of 0.00001 and batch size of 16.
    """

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(3, 64, 7, padding="same")
        self.conv2 = nn.Conv2d(64, 128, 5, padding="same")
        self.conv3 = nn.Conv2d(128, 128, 3, padding="same")
        self.conv4 = nn.Conv2d(128, 256, 3, padding="same")
        self.conv5 = nn.Conv2d(256, 256, 3, padding="same")
        self.fc1 = nn.Linear(256 * 62 * 62, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Example transforms ----------------------------------------------------------

def get_train_transform():
    return Compose([
        ColorJitter(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomErasing(),
        SquarePad(),
        Resize((500, 500)),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

def get_predict_transform():
    return Compose([
        SquarePad(),
        Resize((500, 500)),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

# Example training run --------------------------------------------------------

def train_binary_cnn():

    # Split data
    df = pd.read_csv(DATASET_CSV)
    df = df[["filename", "circle"]].reset_index(drop=True)
    df["filename"] = df["filename"].str.replace(
        "image_", "tests/datasets/shapes/images/image_")
    train_df, val_df, test_df = split_dataframe(df, 0.75, 0.125)

    # Get datasets
    train_dataset = ImagePathDataset(
        train_df,
        read_mode="RGB",
        transform=get_train_transform())

    val_dataset = ImagePathDataset(
        val_df,
        read_mode="RGB",
        transform=get_predict_transform())

    test_dataset = ImagePathDataset(
        test_df,
        read_mode="RGB",
        transform=get_predict_transform())

    # Create model, loss function and optimizer
    model = BinaryCNN()
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Create trainer
    trainer = Trainer(
        model=model,
        model_path=BINARY_CNN_MODEL_PATH,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_func=loss_func,
        optimizer=optimizer,
        metrics=[
            Accuracy(),
            Precision(),
            Recall(),
            F1()],
        best_metric=None)

    # Train
    trainer.train(batch_size=16, epochs=6)

    # Test
    trainer.test(test_dataset, batch_size=16)