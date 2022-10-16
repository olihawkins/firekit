"""
Test image datasets.
"""

# Imports ---------------------------------------------------------------------

import os
import pandas as pd
import torch
import unittest

from torchvision.io import ImageReadMode
from torchvision.io import read_image
from torchvision.transforms import Resize

from firekit.vision import ImagePathDataset

# Constants -------------------------------------------------------------------

SHAPES_DIR = os.path.join("tests", "datasets", "shapes")
IMAGES_DIR = os.path.join(SHAPES_DIR, "images")
LABELS_CSV = os.path.join(SHAPES_DIR, "labels.csv")

# Test data -------------------------------------------------------------------

shapes_df = pd.read_csv(LABELS_CSV).iloc[:, 0:4].reset_index(drop=True)

# Test combine_party_memberships ----------------------------------------------

class CreateImageDataset(unittest.TestCase):

    """
    Test that ImagePathDataset can be created.
    """

    def test_image_path_dataset_is_created_with_dataframe(self):
        ipd = ImagePathDataset(data=shapes_df)
        self.assertTrue(ipd.data.equals(shapes_df))

    def test_image_path_dataset_is_created_without_read_mode(self):
        ipd = ImagePathDataset(data=shapes_df)
        self.assertEqual(ipd.read_mode, ImageReadMode.UNCHANGED)

    def test_image_path_dataset_is_created_with_read_mode_gray(self):
        ipd = ImagePathDataset(data=shapes_df, read_mode="GRAY")
        self.assertEqual(ipd.read_mode, ImageReadMode.GRAY)

    def test_image_path_dataset_is_created_with_read_mode_gray_alpha(self):
        ipd = ImagePathDataset(data=shapes_df, read_mode="GRAY_ALPHA")
        self.assertEqual(ipd.read_mode, ImageReadMode.GRAY_ALPHA)

    def test_image_path_dataset_is_created_with_read_mode_rgb(self):
        ipd = ImagePathDataset(data=shapes_df, read_mode="RGB")
        self.assertEqual(ipd.read_mode, ImageReadMode.RGB)

    def test_image_path_dataset_is_created_with_read_mode_rgb_alpha(self):
        ipd = ImagePathDataset(data=shapes_df, read_mode="RGB_ALPHA")
        self.assertEqual(ipd.read_mode, ImageReadMode.RGB_ALPHA)

    def test_image_path_dataset_is_created_with_transform(self):
        image_path = os.path.join(IMAGES_DIR, "image_1.png")
        image = read_image(image_path, ImageReadMode.RGB).type(torch.float32)
        transform = Resize((256, 256))
        ipd = ImagePathDataset(
            data=shapes_df, 
            transform=transform)
        transformed_image = ipd.transform(image)
        self.assertEqual(transformed_image.size()[1], 256)

    def test_image_path_dataset_is_created_with_target_transform(self):
        labels = torch.tensor([1,0,0], dtype=torch.float32)
        expected_labels = torch.tensor([2,0,0], dtype=torch.float32)
        target_transform = lambda x: x * 2
        ipd = ImagePathDataset(
            data=shapes_df, 
            target_transform=target_transform)
        transformed_labels = ipd.target_transform(labels)
        self.assertTrue(transformed_labels.eq(expected_labels).all())