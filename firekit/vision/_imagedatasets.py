"""
ImageDataset class.
"""

# Imports ---------------------------------------------------------------------

import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode

# ImageReadError --------------------------------------------------------------

class ImageReadError(Exception):
    pass

# ImagePathDataset ------------------------------------------------------------

class ImagePathDataset(Dataset):

    def __init__(
        self, 
        data, 
        read_mode=None,
        transform=None, 
        target_transform=None):

        self.data = data
        self.transform = transform
        self.target_transform = target_transform

        if read_mode == "GRAY":
            self.read_mode = ImageReadMode.GRAY
        elif read_mode == "GRAY_ALPHA":
            self.read_mode = ImageReadMode.GRAY_ALPHA
        elif read_mode == "RGB":
            self.read_mode = ImageReadMode.RGB
        elif read_mode == "RGB_ALPHA":
            self.read_mode = ImageReadMode.RGB_ALPHA
        else:
            self.read_mode = ImageReadMode.UNCHANGED

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        try:

            image_path = self.data.iloc[idx, 0]
            image_labels = self.data.iloc[idx, 1:]
            image = read_image(image_path, self.read_mode).type(torch.float32)
            labels = torch.tensor(image_labels, dtype=torch.float32)
            
            if self.transform:
                image = self.transform(image)
            
            if self.target_transform:
                labels = self.target_transform(label)

            return image, labels

        except Exception as error:
            msg = f"Error reading {image_path}: {error}"
            raise ImageReadError(msg)

    @classmethod
    def create_unlabelled(
        cls,
        dir_path,
        read_mode=None,
        transform=None):

        files = []
        for f in os.listdir(dir_path):
            if not f.startswith("."):
                files.append(os.path.join(dir_path, f))
        data = pd.DataFrame({"path": files, "label": -1})

        return cls(
            data, 
            read_mode=read_mode,
            transform=transform)