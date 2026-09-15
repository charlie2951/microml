# Using decision tree, KNN and SVM classifier
import microml
from array import array

# 1. Real Iris Dataset Samples (4 Features per flower)
# Target Classes: 0 = Setosa, 1 = Versicolor, 2 = Virginica
X_train = array('f', [
    # Setosa (0)
    5.1, 3.5, 1.4, 0.2,
    4.9, 3.0, 1.4, 0.2,
    4.7, 3.2, 1.3, 0.2,
    # Versicolor (1)
    7.0, 3.2, 4.7, 1.4,
    6.4, 3.2, 4.5, 1.5,
    6.9, 3.1, 4.9, 1.5,
    # Virginica (2)
    6.3, 3.3, 6.0, 2.5,
    5.8, 2.7, 5.1, 1.9,
    7.1, 3.0, 5.9, 2.1
])
y_train = array('i', [0, 0, 0, 1, 1, 1, 2, 2, 2])

# Iris Test Dataset
X_test = [
    array('f', [5.0, 3.6, 1.4, 0.2]), # Actual: Setosa (0)
    array('f', [4.6, 3.1, 1.5, 0.2]), # Actual: Setosa (0)
    array('f', [5.5, 2.3, 4.0, 1.3]), # Actual: Versicolor (1)
    array('f', [6.5, 2.8, 4.6, 1.5]), # Actual: Versicolor (1)
    array('f', [6.7, 3.1, 5.6, 2.4]), # Actual: Virginica (2)
    array('f', [6.0, 3.0, 4.8, 1.8])  # Actual: Virginica (2)
]
y_true = [0, 0, 1, 1, 2, 2]

# 2. Train Models (4 Features, 3 Classes)
knn = microml.KNN(3)
knn.fit(X_train, y_train, 4, 3)

dt = microml.DecisionTree(5)
dt.fit(X_train, y_train, 4, 3)

# 3. Formatted Matrix Printer
def print_confusion_matrix(title, y_true, y_pred, labels):
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    print(f"\n--- Confusion Matrix: {title} ---")
    print(f"{'Actual \\ Pred':<15} | {' '.join([f'{l:>10}' for l in labels])}")
    print("-" * (18 + 11 * n))
    for i, row in enumerate(matrix):
        row_str = " ".join([f"{val:>10}" for val in row])
        print(f"{labels[i]:<15} | {row_str}")

# 4. Evaluate Multi-Class Models
iris_classes = ["Setosa", "Versicolor", "Virginica"]

knn_preds = [knn.predict_proba(s)[0] for s in X_test]
print_confusion_matrix("Iris KNN (k=3)", y_true, knn_preds, iris_classes)

dt_preds = [dt.predict(s)[0] for s in X_test]
print_confusion_matrix("Iris DecisionTree", y_true, dt_preds, iris_classes)

# 5. Evaluate Binary SVM (Is Setosa [1] or Not Setosa [0])
binary_y_train = array('i', [1, 1, 1, 0, 0, 0, 0, 0, 0])
binary_y_true = [1, 1, 0, 0, 0, 0]

svm = microml.SVM(microml.KERNEL_RBF, 0.5)
svm.fit(X_train, binary_y_train, 4, 150, 0.01, 1.0)

svm_preds = [svm.predict(s) for s in X_test]
print_confusion_matrix("Iris RBF SVM (Setosa Detector)", binary_y_true, svm_preds, ["Non-Setosa", "Setosa"])
