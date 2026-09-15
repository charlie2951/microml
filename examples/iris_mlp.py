# IRIS flower classification
import array
import microml

# 1. Multi-Class Dataset (6 training samples, 4 features)
X_train = array.array('f', [
    5.1, 3.5, 1.4, 0.2, # Setosa (0)
    4.9, 3.0, 1.4, 0.2, # Setosa (0)
    7.0, 3.2, 4.7, 1.4, # Versicolor (1)
    6.4, 3.2, 4.5, 1.5, # Versicolor (1)
    6.3, 3.3, 6.0, 2.5, # Virginica (2)
    5.8, 2.7, 5.1, 1.9  # Virginica (2)
])
y_train = array.array('i', [0, 0, 1, 1, 2, 2])

# 2. Train MLP Model (4 inputs, 8 hidden, 3 outputs)
nn = microml.MLP(4, 8, 3)
nn.fit(X_train, y_train, 3000, 0.01, 0.9)

# 3. Test Set (Ground Truth vs Model Predictions)
X_test = [
    array.array('f', [5.0, 3.4, 1.5, 0.2]), # True: 0
    array.array('f', [4.8, 3.1, 1.6, 0.2]), # True: 0
    array.array('f', [6.2, 2.9, 4.3, 1.3]), # True: 1
    array.array('f', [5.9, 3.0, 4.2, 1.5]), # True: 1
    array.array('f', [6.5, 3.0, 5.2, 2.0]), # True: 2
    array.array('f', [5.7, 2.8, 4.5, 1.3])  # True: 1 (Edge case)
]
y_true = [0, 0, 1, 1, 2, 2]

# Collect Model Predictions
y_pred = []
for sample in X_test:
    label, _ = nn.predict(sample)
    y_pred.append(label)

# 4. Generate and Print Confusion Matrix
num_classes = 3
# Initialize a 3x3 matrix filled with zeroes
matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

# Populate matrix: rows = Actual, columns = Predicted
for true_label, pred_label in zip(y_true, y_pred):
    matrix[true_label][pred_label] += 1

# Print Formatted Table
class_labels = ["Setosa", "Versicolor", "Virginica"]

print("\n--- Confusion Matrix ---")
print(f"{'Actual \\ Pred':<15} | {' '.join([f'{lbl:>10}' for lbl in class_labels])}")
print("-" * 52)

for i, row in enumerate(matrix):
    row_str = " ".join([f"{count:>10}" for count in row])
    print(f"{class_labels[i]:<15} | {row_str}")
