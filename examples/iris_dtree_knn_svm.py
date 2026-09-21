# Using decision tree, KNN and SVM classifier
import microml
from array import array
from ulab import numpy as np
# 1. Real Iris Dataset Samples (4 Features per flower)
# Target Classes: 0 = Setosa, 1 = Versicolor, 2 = Virginica
X_train = np.array([
    # Setosa (0)
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    # Versicolor (1)
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    # Virginica (2)
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1]
])
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# Iris Test Dataset
X_test =np.array([
       [5.0, 3.6, 1.4, 0.2], # Actual: Setosa (0)
       [4.6, 3.1, 1.5, 0.2], # Actual: Setosa (0)
       [5.5, 2.3, 4.0, 1.3], # Actual: Versicolor (1)
       [6.5, 2.8, 4.6, 1.5], # Actual: Versicolor (1)
       [6.7, 3.1, 5.6, 2.4], # Actual: Virginica (2)
       [6.0, 3.0, 4.8, 1.8]  # Actual: Virginica (2)
])
y_true = np.array([0, 0, 1, 1, 2, 2])

# 2. Train Models (4 Features, 3 Classes)
knn = microml.KNN(3)
knn.fit(X_train, y_train, 4, 3)
dt = microml.DecisionTree(5)
dt.fit(X_train, y_train, 4, 3)

from microml import confusion_matrix, classification_report
# 4. Evaluate Multi-Class Models
iris_classes = ["Setosa", "Versicolor", "Virginica"]

knn_preds = np.array([knn.predict_proba(s)[0] for s in X_test])
confusion_matrix(y_true, knn_preds, num_classes=3, labels=iris_classes, title="Iris KNN (k=3)")
classification_report(y_true, knn_preds, labels=iris_classes, digits=4)

dt_preds = np.array([dt.predict(s)[0] for s in X_test])
confusion_matrix( y_true, dt_preds,  num_classes=3, labels=iris_classes,title="Iris DecisionTree")
classification_report(y_true, dt_preds, labels=iris_classes, digits=4)
# 5. Evaluate Binary SVM (Is Setosa [1] or Not Setosa [0])
binary_y_train = array('i', [1, 1, 1, 0, 0, 0, 0, 0, 0])
binary_y_true = [1, 1, 0, 0, 0, 0]

svm = microml.SVM(microml.KERNEL_RBF, 0.5)
svm.fit(X_train, binary_y_train, 4, 150, 0.01, 1.0)

svm_preds = np.array([svm.predict(s) for s in X_test])
confusion_matrix( binary_y_true, svm_preds,  num_classes=2, labels=["Non-Setosa", "Setosa"],title="Iris RBF SVM (Setosa Detector)")
classification_report(binary_y_true, svm_preds, labels=["Non-Setosa", "Setosa"], digits=4)
