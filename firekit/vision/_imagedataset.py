"""
ImageDataset class.
"""

# Imports ---------------------------------------------------------------------

import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io import ImageReadMode

# Dataset ---------------------------------------------------------------------

class ImageDataset(Dataset):

    def __init__(
        self, 
        data, 
        transform=None, 
        target_transform=None):

        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        image_path = self.data.iloc[idx, 0]
        image_label = self.data.iloc[idx, 1]
        image = read_image(image_path, ImageReadMode.RGB).type(torch.float32)
        label = torch.tensor(image_label, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
