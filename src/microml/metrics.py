import ulab
from ulab import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculates accuracy score between true labels and predicted labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
            
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Computes confusion matrix to evaluate classification accuracy.
    Rows represent true labels, columns represent predicted labels.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Auto-detect number of classes if not specified
    if num_classes is None:
        max_true = int(np.max(y_true))
        max_pred = int(np.max(y_pred))
        num_classes = max(max_true, max_pred) + 1

    # Initialize N x N matrix with zeros
    cm = np.zeros((num_classes, num_classes), dtype=np.int16)

    for i in range(len(y_true)):
        actual = int(y_true[i])
        predicted = int(y_pred[i])
        if 0 <= actual < num_classes and 0 <= predicted < num_classes:
            cm[actual, predicted] += 1

    return cm
