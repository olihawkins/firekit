"""
Utility functions.
"""

# Imports ---------------------------------------------------------------------

import numpy as np
import warnings

# Utility functions -----------------------------------------------------------

def sigmoid(logits):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1 / (1 + np.exp(-1 * logits))