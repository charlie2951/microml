import microml
from array import array

# Data setup: 3 Classes, 2 Features
X = array('f', [
    1.0, 2.0,   2.1, 2.8,   # Class 0
   -1.0, -2.0, -1.2, -1.8,  # Class 1
    5.0, -5.0,  5.2, -4.8   # Class 2
])
y = array('i', [0, 0, 1, 1, 2, 2])
sample = array('f', [1.1, 2.2])

# 1. Multi-class KNN with probabilities(k=3)
knn = microml.KNN(3)
#knn.fit(X, y, n_features=2, n_classes=3)
knn.fit(X, y, 2, 3)
label, probs = knn.predict_proba(sample)
print("KNN Class:", label, "Probabilities:", probs)

# 2. Decision Tree with Save / Load Persistence, depth=4
dt = microml.DecisionTree(4)
#dt.fit(X, y, n_features=2, n_classes=3)
dt.fit(X, y, 2, 3)
label, conf = dt.predict(sample)
print("Tree Class:", label, "Confidence:", conf)

# Save binary tree model to microcontroller flash
dt.save("model.bin")

# Load model into new instance
new_dt = microml.DecisionTree()
new_dt.load("model.bin")
print("Loaded Model Predict:", new_dt.predict(sample))

# 3. Non-linear RBF Kernel SVM, gamma=0.5
svm = microml.SVM(microml.KERNEL_RBF, 0.5)
binary_y = array('i', [1, 1, 0, 0, 0, 0])  # Binary classification for SVM
#svm.fit(X, binary_y, n_features=2, epochs=150, lr=0.01, C=1.0)
svm.fit(X, binary_y, 2, 150, 0.01, 1.0)
print("RBF SVM Predict:", svm.predict(sample))
