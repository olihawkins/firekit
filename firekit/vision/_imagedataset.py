"""
ImageDataset class.
"""

# Imports ---------------------------------------------------------------------

import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode

# ImageReadError ----------------------------------------------------------------

class ImageReadError(Exception):
    pass

# ImageDataset ----------------------------------------------------------------

class ImageDataset(Dataset):

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
            image_label = self.data.iloc[idx, 1]

            image = read_image(image_path, self.read_mode).type(torch.float32)
            label = torch.tensor(image_label, dtype=torch.float32).unsqueeze(0)
            
            if self.transform:
                image = self.transform(image)
            
            if self.target_transform:
                label = self.target_transform(label)

            return image, label

        except Exception:
            msg = f"Error reading {image_path}"
            raise ImageReadError(msg)
