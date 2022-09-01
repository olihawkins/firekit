"""
Metrics for use during training.
"""

# Imports ---------------------------------------------------------------------

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from firekit.utils import sigmoid

# Base metric class -----------------------------------------------------------

class Metric:

    """
    Base class for metrics.
    """

    def __init__(self, precision):
        self.name = "metric"
        self.label = "Metric"
        self.precision = precision

    def get_metric(self, targets, predictions):
        pass

    def get_metric_loss(self, targets, predictions):
        pass

    def get_formatted_metric(self, targets, predictions):
        metric = self.get_metric(targets, predictions)
        return f"{metric:.{self.precision}f}"

    def get_reported_metric(self, targets, predictions):
        formatted_metric = self.get_formatted_metric(targets, predictions)
        return f"{self.label}: {formatted_metric}"

# Classification metric class -------------------------------------------------

class ClassificationMetric(Metric):

    """
    Base class for classification metrics.
    """

    def __init__(self, precision, logits):
        super().__init__(precision)
        self.logits = logits

    def get_metric_loss(self, targets, predictions):
        return 1. - self.get_metric(targets, predictions)

# Accuracy metric class -------------------------------------------------------

class Accuracy(ClassificationMetric):

    """
    Accuracy metric class.
    """

    def __init__(self, precision=4, logits=True):
        super().__init__(precision, logits)
        self.name = "accuracy"
        self.label = "Accuracy"

    def get_metric(self, targets, predictions):
        if self.logits == True:
            predictions = sigmoid(predictions)
        predictions = np.round(predictions)
        return accuracy_score(targets.flatten(), predictions.flatten())

# Subset accuracy metric class ------------------------------------------------

class SubsetAccuracy(ClassificationMetric):

    """
    Subset accuracy metric class.
    """

    def __init__(self, precision=4, logits=True):
        super().__init__(precision, logits)
        self.name = "accuracy"
        self.label = "Subset accuracy"

    def get_metric(self, targets, predictions):
        if self.logits == True:
            predictions = sigmoid(predictions)
        predictions = np.round(predictions)
        return accuracy_score(targets, predictions)

# Precision metric class ------------------------------------------------------

class Precision(ClassificationMetric):

    """
    Precision metric class.
    """

    def __init__(self, precision=4, logits=True):
        super().__init__(precision, logits)
        self.name = "precision"
        self.label = "Precision"

    def get_metric(self, targets, predictions):
        if self.logits == True:
            predictions = sigmoid(predictions)
        predictions = np.round(predictions)
        return precision_score(targets, predictions)

# Recall metric class ---------------------------------------------------------

class Recall(ClassificationMetric):

    """
    Recall metric class.
    """

    def __init__(self, precision=4, logits=True):
        super().__init__(precision, logits)
        self.name = "recall"
        self.label = "Recall"

    def get_metric(self, targets, predictions):
        if self.logits == True:
            predictions = sigmoid(predictions)
        predictions = np.round(predictions)
        return recall_score(targets, predictions)

# Recall metric class ---------------------------------------------------------

class F1(ClassificationMetric):

    """
    F1 score metric class.
    """

    def __init__(self, precision=4, logits=True):
        super().__init__(precision, logits)
        self.name = "f1_score"
        self.label = "F1 score"

    def get_metric(self, targets, predictions):
        if self.logits == True:
            predictions = sigmoid(predictions)
        predictions = np.round(predictions)
        return f1_score(targets, predictions)