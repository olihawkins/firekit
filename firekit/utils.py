"""
Utility functions.
"""

# Imports ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Utility functions -----------------------------------------------------------

def sigmoid(logits):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1 / (1 + np.exp(-1 * logits))