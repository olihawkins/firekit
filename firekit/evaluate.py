"""
Functions for evaluating model performance.
"""

# Imports ---------------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from firekit.utils import sigmoid

# Evaluation functions --------------------------------------------------------

def evaluate_binary_classification(targets, predictions, logits=True):

    # Convert logits
    if logits == True:
        predictions = sigmoid(predictions)

    # Convert predictions to binary class labels
    predictions = np.round(predictions)

    # Get metrics
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    
    metrics = {
        "accuracy": accuracy, 
        "precision": precision,
        "recall": recall,
        "f1": f1}

    return metrics

def evaluate_multilabel_classification(targets, predictions, labels, logits=True):

    # Convert logits
    if logits == True:
        predictions = sigmoid(predictions)

    # Convert logit predictions to binary class labels
    predictions = np.round(predictions)

    # Get metrics
    accuracy = accuracy_score(targets.flatten(), predictions.flatten())
    subset_accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average=None)
    recall = recall_score(targets, predictions, average=None)
    f1 = f1_score(targets, predictions, average=None)
    
    metrics = {
        "accuracy": accuracy,
        "subset_accuracy": subset_accuracy, 
        "precision": precision,
        "recall": recall,
        "f1": f1}

    summary = np.stack((precision, recall, f1)).transpose()
    summary = pd.DataFrame(summary, columns=["precision", "recall", "f1"])
    summary.insert(0, "label", labels)
    
    return metrics, summary