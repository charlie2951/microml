import array
import microml

# 1. Initialize a 3-layer MLP: 2 Inputs -> 4 Hidden Neurons -> 2 Output Classes (0 or 1)
# Architecture: [2, 4, 2], is_regression=False
mlp = microml.MLP([2, 4,4,2], False)

# 2. Prepare XOR Training Dataset (4 samples, 2 features each)
X_data = array.array('f', [
    0.0, 0.0,  # Sample 0 -> Output 0
    0.0, 1.0,  # Sample 1 -> Output 1
    1.0, 0.0,  # Sample 2 -> Output 1
    1.0, 1.0   # Sample 3 -> Output 0
])

# Target labels (0 or 1) stored as float array
y_data = array.array('f', [0.0, 1.0, 1.0, 0.0])

# 3. Train the Model
# Parameters: fit(X, y, epochs=2000, learning_rate=0.05, momentum=0.9)
print("Training XOR network...")
mlp.fit(X_data, y_data, 10000, 0.001, 0.9)

# 4. Evaluate Predictions
print("\n--- XOR Predictions ---")
test_inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

for sample in test_inputs:
    x_test = array.array('f', sample)
    pred_class = mlp.predict(x_test)
    probs = mlp.predict_proba(x_test)
    
    print("Input: {} -> Predicted Class: {} (Probabilities: Class 0: {:.2f}%, Class 1: {:.2f}%)".format(
        sample, pred_class, probs[0] * 100, probs[1] * 100
    ))
