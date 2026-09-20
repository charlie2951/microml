# IRIS flower classification using MLP model
import array
import microml
from ulab import numpy as np
# 1. Multi-Class Dataset (6 training samples, 4 features)
X_train = np.array( [
    [5.1, 3.5, 1.4, 0.2], # Setosa (0)
    [4.9, 3.0, 1.4, 0.2], # Setosa (0)
    [7.0, 3.2, 4.7, 1.4], # Versicolor (1)
    [6.4, 3.2, 4.5, 1.5], # Versicolor (1)
    [6.3, 3.3, 6.0, 2.5], # Virginica (2)
    [5.8, 2.7, 5.1, 1.9]  # Virginica (2)
])
y_train = np.array([0, 0, 1, 1, 2, 2])

# 2. Train MLP Model (4 inputs, 8 hidden, 3 outputs)
nn = microml.MLP([4, 8, 8, 3], False)
nn.fit(X_train, y_train, 5000, 0.005, 0.9)

# 3. Test Set (Ground Truth vs Model Predictions)
X_test = np.array([
    [5.0, 3.4, 1.5, 0.2], # True: 0
    [4.8, 3.1, 1.6, 0.2], # True: 0
    [6.2, 2.9, 4.3, 1.3], # True: 1
    [5.9, 3.0, 4.2, 1.5], # True: 1
    [6.5, 3.0, 5.2, 2.0], # True: 2
    [5.7, 2.8, 4.5, 1.3]  # True: 1 (Edge case)
])
y_true = np.array([0, 0, 1, 1, 2, 2])

# Collect Model Predictions
y_pred = []
for sample in X_test:
    label= nn.predict(sample)
    y_pred.append(label)

y_p=np.array(y_pred)
#print report and confusion matrix
labels = ["Setosa", "Versicolor", "Virginica"]
microml.confusion_matrix(y_true, y_p, num_classes=3, labels=labels, title="IRIS flower classification")
microml.classification_report(y_true, y_p, labels=labels, digits=4)
