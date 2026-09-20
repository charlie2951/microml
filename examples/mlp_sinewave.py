import math
from ulab import numpy as np
import microml

# 1. Generate Training Data (Sinewave: y = sin(x))
NUM_SAMPLES = 50

# Generate evenly spaced x points in range [-pi, pi]
X_train = np.linspace(-math.pi, math.pi, NUM_SAMPLES)

# Compute y values using vectorized numpy sine function
y_train = np.sin(X_train)

# 2. Instantiate MLPRegressor
# Architecture: 1 Input -> 16 Hidden Neurons -> 8 Hidden Neurons -> 1 Output
mlp_reg = microml.MLP([1, 16, 8, 1], True)

# 3. Train the Model
print("Training MLP Regressor on Sine Wave...")
mlp_reg.fit(X_train, y_train, 1000, 0.001, 0.9)

# 4. Evaluate & Predict
print("\nPredictions vs Actual:")
print("---------------------------------------")
print("  x    |  Predicted  |   Actual   | Error")
print("---------------------------------------")

# Test 5 sample points
#test_points = np.array([-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2])
test_points=np.linspace(-math.pi, math.pi, 20)
for x_test in test_points:
    # Wrap test scalar in a 1D ulab numpy array
    x_buf = np.array([x_test])

    # Unpack predicted float from returned ulab array or list
    pred = mlp_reg.predict(x_buf)[0]
    actual = math.sin(x_test)
    error = abs(pred - actual)

    print("{: .2f} |  {: .4f}    |  {: .4f}  | {:.4f}".format(
        x_test, pred, actual, error
    ))
