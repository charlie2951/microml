import array
import microml

# 4 samples, 2 input features (XOR problem)
X = array.array('f', [0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  1.0, 1.0])
y = array.array('i', [0, 1, 1, 0])

# Initialize MLP: 2 Inputs, 4 Hidden Units, 2 Output Classes
nn = microml.MLP(2, 4, 2)

# Fit model: X, y, epochs=500, learning_rate=0.1, momentum=0.9
nn.fit(X, y, 5000, 0.001, 0.9)

# Predict single sample
test_sample = array.array('f', [1.0, 0.0])
label, probs = nn.predict(test_sample)

print("Predicted Class:", label)
print("Class Probabilities:", probs)
