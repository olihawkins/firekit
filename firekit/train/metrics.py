"""
Metrics for use during training.
"""

# Imports ---------------------------------------------------------------------

import numpy as np

from sklearn.metrics import accuracy_score

from firekit.utils import sigmoid

# Base metric class -----------------------------------------------------------

class Metric:

    """
    Base class for metrics.
    """

    def __init__(self):
        self.name = "metric"
        self.label = "Metric"
        self.precision = 2

    def get_metric(self, targets, predictions):
        pass

    def get_loss_metric(self, targets, predictions):
        pass

    def get_formatted_metric(self, targets, predictions):
        metric = self.get_metric(targets, predictions)
        return f"{metric:.{self.precision}f}"

    def get_reported_metric(self, targets, predictions):
        formatted_metric = self.get_formatted_metric(targets, predictions)
        return f"{self.label}: {formatted_metric}"

# Accuracy metric class -------------------------------------------------------

class Accuracy(Metric):

    """
    Accuracy metric class.
    """

    def __init__(self, logits=True):
        self.name = "accuracy"
        self.label = "Accuracy"
        self.precision = 4
        self.logits = logits

    def get_metric(self, targets, predictions):
        if self.logits == True:
            predictions = sigmoid(predictions)
        predictions = np.round(predictions)
        return accuracy_score(targets.flatten(), predictions.flatten())

    def get_loss_metric(self, targets, predictions):
        return 1. - self.get_metric(targets, predictions)

# Subset accuracy metric class ------------------------------------------------

class SubsetAccuracy(Metric):

    """
    Subset accuracy metric class.
    """

    def __init__(self, logits=True):
        self.name = "accuracy"
        self.label = "Accuracy"
        self.precision = 4
        self.logits = logits

    def get_metric(self, targets, predictions):
        if self.logits == True:
            predictions = sigmoid(predictions)
        predictions = np.round(predictions)
        return accuracy_score(targets, predictions)

    def get_loss_metric(self, targets, predictions):
        return 1. - self.get_metric(targets, predictions)