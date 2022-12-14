"""
This file contains unit tests for the Trainer class and integration tests that 
run the Trainer with a known model and dataset.
"""

# Imports ---------------------------------------------------------------------

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

from firekit.preprocess import split_pandas
from firekit.train import Trainer
from firekit.train.metrics import BinaryAccuracy
from firekit.train.metrics import BinaryPrecision
from firekit.train.metrics import BinaryRecall
from firekit.train.metrics import BinaryF1
from firekit.train.metrics import MulticlassAccuracy
from firekit.vision import ImagePathDataset
from firekit.vision import MulticlassImagePathDataset
from firekit.vision.transforms import SquarePad

# Constants -------------------------------------------------------------------

SHAPES_DIR = os.path.join("tests", "datasets", "shapes")
IMAGES_DIR = os.path.join(SHAPES_DIR, "images")
DATASET_CSV = os.path.join(SHAPES_DIR, "labels.csv")
BINARY_CNN_PATH = os.path.join("tests", "models", "binary-cnn.pt")
MULTICLASS_CNN_PATH = os.path.join("tests", "models", "multiclass-cnn.pt")

# Models ----------------------------------------------------------------------

class CNN(nn.Module):

    """
    A CNN classifier.
    """

    def __init__(self, outputs, last_pool_size=16):
        super().__init__()
        self.last_pool_size = last_pool_size
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(3, 64, 7, padding="same")
        self.conv2 = nn.Conv2d(64, 128, 5, padding="same")
        self.conv3 = nn.Conv2d(128, 128, 3, padding="same")
        self.conv4 = nn.Conv2d(128, 256, 3, padding="same")
        self.conv5 = nn.Conv2d(256, 256, 3, padding="same")
        self.fc1 = nn.Linear(256 * last_pool_size * last_pool_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, self.last_pool_size)
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
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
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

# Train binary CNN classifier -------------------------------------------------

def train_binary_cnn(epochs=3):

    # Prepare data
    df = pd.read_csv(DATASET_CSV)
    df = df[["filename", "square"]]
    train_df, val_df, test_df = split_pandas(df, 0.75, 0.125)

    # Get datasets
    train_dataset = ImagePathDataset(
        data = train_df,
        read_mode="RGB",
        transform=get_train_transform())

    val_dataset = ImagePathDataset(
        data = val_df,
        read_mode="RGB",
        transform=get_predict_transform())

    test_dataset = ImagePathDataset(
        data = test_df,
        read_mode="RGB",
        transform=get_predict_transform())

    # Create model, loss function and optimizer
    model = CNN(outputs=1)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Create trainer
    trainer = Trainer(
        model=model,
        model_path=BINARY_CNN_PATH,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_func=loss_func,
        optimizer=optimizer,
        metrics=[
            BinaryAccuracy(),
            BinaryPrecision(),
            BinaryRecall(),
            BinaryF1()],
        best_metric=None)

    # Train
    trainer.train(batch_size=16, epochs=epochs)

    # Test
    trainer.test(test_dataset, batch_size=16)

# Train multiclass CNN classifier ---------------------------------------------

def train_multiclass_cnn(epochs=3):

    # Prepare data
    df = pd.read_csv(DATASET_CSV)
    df = df[df["square"] == 1]
    argmax = lambda row: np.argmax(row)
    df["color"] = df[["red", "green", "blue"]].apply(argmax, axis=1)
    df = df[["filename", "color"]]
    train_df, val_df, test_df = split_pandas(df, 0.75, 0.125)

    # Get datasets
    train_dataset = MulticlassImagePathDataset(
        data=train_df,
        read_mode="RGB",
        transform=get_train_transform())

    val_dataset = MulticlassImagePathDataset(
        data=val_df,
        read_mode="RGB",
        transform=get_predict_transform())

    test_dataset = MulticlassImagePathDataset(
        data=test_df,
        read_mode="RGB",
        transform=get_predict_transform())

    # Create model, loss function and optimizer
    model = CNN(outputs=3)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Create trainer
    trainer = Trainer(
        model=model,
        model_path=MULTICLASS_CNN_PATH,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_func=loss_func,
        optimizer=optimizer,
        metrics=[MulticlassAccuracy()],
        best_metric=None)

    # Train
    trainer.train(batch_size=16, epochs=epochs)

    # Test
    trainer.test(test_dataset, batch_size=16)