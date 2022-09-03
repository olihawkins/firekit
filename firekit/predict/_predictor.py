"""
Predictor classes.
"""

# Imports ---------------------------------------------------------------------

import numpy as np
import torch

from torch.utils.data import DataLoader

from firekit.utils import get_device

# Predictor class ---------------------------------------------------------------

class Predictor():

    def __init__(
        self, 
        model,
        device=None):
        
        # Set up device
        self.device = torch.device(get_device(device))

        # Set up model
        self.model = model
        self.model.eval()
        self.model.to(self.device)
        
    def predict(self, dataset, batch_size):
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False)
        predictions = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())
        predictions = np.concatenate(predictions)
        return predictions


