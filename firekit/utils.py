"""
Utility functions.
"""

# Imports ---------------------------------------------------------------------

import numpy as np
import torch
import warnings

# Utility functions -----------------------------------------------------------

def sigmoid(logits):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1 / (1 + np.exp(-1 * logits))

def get_device(device=None):
    if device == None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return device