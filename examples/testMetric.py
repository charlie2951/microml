import ulab
from ulab import numpy as np
import microml

y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 0, 0, 1, 2, 1, 2, 2, 1])  # 2 misclassifications

labels = ["Setosa", "Versicolor", "Virginica"]

# Print combined performance metrics
#microml.classification_report(y_true, y_pred, labels=labels)
microml.confusion_matrix(y_true, y_pred, num_classes=3, labels=labels, title="IRIS flower classification")
microml.classification_report(y_true, y_pred, labels=labels, digits=4)

