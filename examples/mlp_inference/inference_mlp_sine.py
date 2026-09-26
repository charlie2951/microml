import math
from ulab import numpy as np
import microml

test_points=np.linspace(-math.pi, math.pi, 20)

# Load model weights into an existing instance
saved_model = microml.MLP([2, 1])  # dummy model, Dimensions will be updated on load
saved_model.load("sine_mlp.bin")
test_points=np.linspace(-math.pi, math.pi, 50)
# 4. Evaluate & Predict
print("\nSine wave Regression problem: Predictions vs Actual:")
print("---------------------------------------")
print("  x    |  Predicted  |   Actual   | Error")
print("---------------------------------------")

for x_test in test_points:
    # Wrap test scalar in a 1D ulab numpy array
    x_buf = np.array([x_test])

    # Unpack predicted float from returned ulab array or list
    pred = saved_model.predict(x_buf)[0]
    actual = math.sin(x_test)
    error = abs(pred - actual)
    
    print("{: .2f} |  {: .4f}    |  {: .4f}  | {:.5f}".format(
        x_test, pred, actual, error
    ))
