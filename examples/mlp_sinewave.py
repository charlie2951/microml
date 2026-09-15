import array
import math
import microml

# 1. Generate Training Data (Sinewave: y = sin(x))
NUM_SAMPLES = 50
X_raw = []
y_raw = []

# Generate points in range [-pi, pi]
step = (2 * math.pi) / NUM_SAMPLES
for i in range(NUM_SAMPLES):
    x_val = -math.pi + i * step
    y_val = math.sin(x_val)

    X_raw.append(x_val)
    y_raw.append(y_val)

# Pack data into C-compatible float arrays
X_train = array.array("f", X_raw)
y_train = array.array("f", y_raw)

# 2. Instantiate MLPRegressor
# Architecture: 1 Input -> 16 Hidden Neurons -> 1 Output
mlp_reg = microml.MLP([1, 16, 8, 1], True)

# 3. Train the Model
print("Training MLP Regressor on Sine Wave...")
mlp_reg.fit(X_train, y_train, 5000, 0.001, 0.9)

# 4. Evaluate & Predict
print("\nPredictions vs Actual:")
print("---------------------------------------")
print("  x    |  Predicted  |   Actual   | Error")
print("---------------------------------------")

# Test 5 sample points
test_points = [-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2]

for x_test in test_points:
    x_buf = array.array("f", [x_test])

    # Unpack predicted float from returned list [pred]
    pred = mlp_reg.predict(x_buf)[0]
    actual = math.sin(x_test)
    error = abs(pred - actual)

    print("{: .2f} |  {: .4f}    |  {: .4f}  | {:.4f}".format(
        x_test, pred, actual, error
    ))
