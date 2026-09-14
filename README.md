# MicroML: Lightweight Machine Learning C-Module for MicroPython

`MicroML` is a lightweight, zero-dependency C-module built for MicroPython that enables on-device training and inference directly on microcontrollers (e.g., ESP32, RP2040, STM32, nRF52).

It features classical ML algorithms as well as an optimized **Multi-Layer Perceptron (MLP)** engine supporting non-linear regression using Mean Squared Error (MSE) loss and Stochastic Gradient Descent (SGD) with momentum.

---

## Features

* **Multi-Layer Perceptron (MLP):**
  * Flexible 3-layer architecture (Input $\to$ Hidden $\to$ Output).
  * ReLU activation for hidden layers and Linear activation for continuous output regression.
  * Backpropagation engine powered by SGD with momentum acceleration.
  * Real-time training loss logging outputted directly to the MicroPython REPL using `mp_printf`.
* **Classical ML Algorithms:**
  * **K-Nearest Neighbors (KNN)** with confidence probabilities.
  * **Decision Tree** with Gini impurity splitting and binary `save`/`load` file serialization.
  * **Support Vector Machine (SVM)** supporting both Linear and Radial Basis Function (RBF) kernels.
* **Embedded Resource-Friendly:**
  * Zero dynamic memory fragmentation via MicroPython GC heap routines (`m_new`, `m_free`).
  * Direct buffer protocol execution on native Python `array.array('f')` and `array.array('i')` data types without extra memory copying.

---

## Repository Structure

Place the source files in your custom module directory inside the MicroPython source tree:

```text
microml/
├── microml.c       # MicroPython bindings & C-module wrappers
├── microml.h       # Headers for KNN, DecisionTree, and SVM implementations
├── mlp.c           # MLP Forward/Backward propagation engine
├── mlp.h           # MLP model definitions and memory interfaces
├── micropython.mk  # Makefile configuration
└── CMakeLists.txt  # CMake configuration
```
## Build & Compilation Guide
1. Build System Setup (common for all ports) <p>
* Create a directory and Clone the repo
  ```
  mkdir test_build
  cd test_build
  git clone https://github.com/charlie2951/microml.git
  git clone https://github.com/micropython/micropython.git
  cd micropython
  make -C mpy-cross

  ```
2. Installing Toolchain <p>
Go back to your created build directory and install toolchain<p>
   For esp32 port:
   ```
   git clone https://github.com/espressif/esp-idf.git
   cd esp-idf
   git checkout v5.5.1
   git submodule update --init --recursive
   cd esp-idf
   ./install.sh
   source export.sh # You will need to source export.sh for every new session.
   ```

3. Build your BOARD <p>
  ```
  cd ports/esp32
  make submodules
  make BOARD=ESP32_GENERIC BOARD_VARIANT=SPIRAM  USER_C_MODULES=../../../../microml/src/  #Find board name in ./boards
  ```

Upon successful build, the `firmware.bin` will be available inside `/ports/esp32/build_dir`


## Python API Reference
1. Multi-Layer Perceptron Regressor `microml.MLP` Constructor: <p>
- `microml.MLP(input_dim, hidden_dim, output_dim)`: Allocates model weights and momentum buffers on the MicroPython heap with Xavier uniform random initialization.
- `fit(X, y, epochs=100, lr=0.01, momentum=0.9)`:Trains the neural network using SGD with momentum and MSE loss.
- `X: array.array('f')` containing flattened float inputs (length: $N \times input~dim$).
- `y: array.array('f')` containing flattened float target values (length: $N \times output~dim$).
- `epochs (int)`: Number of training iterations.
- `lr (float)`: Learning rate step size.
- `momentum (float`: Momentum fraction (range: 0.0 to 1.0).
- `predict(X_sample)`Runs forward propagation on a single sample.
- `X_sample:` array.array('f') containing input features for 1 sample. Returns: list of float predicted values.
<p>
 
2. Decision Tree Classifier `(microml.DecisionTree)`:<p>
 
 - `microml.DecisionTree(max_depth=5)`: Instantiates a binary decision tree.
 - `fit(X, y, n_features, n_classes=2)`: Fits tree nodes on integer target labels (array.array('i')).
 - `predict(X_sample)`: Returns a tuple (predicted_class, confidence_score).
 - `save(filename)`: Saves binary tree structure directly to the device filesystem.
 - `load(filename)`: Restores saved model structure from file.

3. K-Nearest Neighbors Classifier `(microml.KNN)`:<p>

- `microml.KNN(k=3)`: Instantiates a KNN classifier.
- `fit(X, y, n_features, n_classes=2)`: Stores dataset pointers on the heap.
- `predict_proba(X_sample)`: Returns a tuple (predicted_class, [class_probabilities]).

4. Support Vector Machine (microml.SVM): <p>

- `microml.SVM(kernel_type, gamma=0.5)`:
- `Kernels`: **microml.KERNEL_LINEAR**, **microml.KERNEL_RBF.fit(X, y, n_features, epochs=100, lr=0.01, C=1.0)**: Optimizes support vector margins.
- `predict(X_sample)`: Returns binary predicted class (0 or 1).

## Sine wave example
```python
import array
import math
import microml

# 1. Prepare Training Dataset (32 Points around a Sine Wave)
N_SAMPLES = 32
raw_X = []
raw_y = []

for i in range(N_SAMPLES):
    angle = (2.0 * math.pi * i) / N_SAMPLES
    
    # Input feature: Normalized Angle in range [0.0, 1.0]
    raw_X.append(angle / (2.0 * math.pi))
    
    # Target value: Continuous Sine in range [-1.0, 1.0]
    raw_y.append(math.sin(angle))

# Pack data into C-compatible float arrays
X = array.array('f', raw_X)
y = array.array('f', raw_y)

# 2. Instantiate MLP Regressor (1 Input -> 12 Hidden Units -> 1 Continuous Output)
nn = microml.MLP(1, 12, 1)

print("Training MLP Sine Wave Regressor...")
# Train model for 2000 epochs with lr=0.05 and momentum=0.9
nn.fit(X, y, 2000, 0.05, 0.9)

# 3. Test Predictions Across Samples
print("\n--- Model Inference Evaluation ---")
test_angles = [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]

for angle in test_angles:
    # Scale test input matching training feature space
    norm_angle = angle / (2.0 * math.pi)
    test_sample = array.array('f', [norm_angle])
    
    # Predict continuous value
    pred = nn.predict(test_sample)
    actual = math.sin(angle)
    
    print("Angle: {:5.2f} rad | Predicted: {:6.3f} | Actual: {:6.3f} | Error: {:6.3f}".format(
        angle, pred[0], actual, abs(pred[0] - actual)
    ))
```
### Result
