![](https://komarev.com/ghpvc/?username=charlie2951&color=brightgreen) ![GitHub commit activity](https://img.shields.io/github/commit-activity/t/charlie2951/microml?label=Total%20commit) ![GitHub Repo stars](https://img.shields.io/github/stars/charlie2951/microml?style=flat)


# MicroML: Lightweight Machine Learning Module for MicroPython

`MicroML` is a lightweight, zero-dependency module built for MicroPython that enables on-device training and inference directly on microcontrollers (e.g., ESP32, RP2040, STM32, nRF52).

It features classical ML algorithms as well as an optimized **Multi-Layer Perceptron (MLP)** engine supporting non-linear regression using Mean Squared Error (MSE) loss and Stochastic Gradient Descent (SGD) with momentum. To use it, you need to use the custom firmware where these API are written in C, compiled as external C modules, and frozen inside the firmware, thus providing optimum performance

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

```text
microml/
├── examples/       # MicroPython bindings & C-module wrappers
├── firmware/       # pre-built firmware for some port
├── src/           # C-module wrappers
       ├──microml.c       # MicroPython bindings & C-module wrappers
       ├── microml.h       # Headers for KNN, DecisionTree, and SVM implementations
       ├── mlp.c           # MLP Forward/Backward propagation engine
       ├── mlp.h           # MLP model definitions and memory interfaces
       ├── micropython.mk  # Makefile configuration
       └── CMakeLists.txt  # CMake configuration
```


## Build & Compilation Guide
1. Typical Build System Setup on an Ubuntu system (common for all ports) <p>

* Install prerequisites
  
    ```
    sudo apt-get update
    sudo apt-get install -y git wget make libncurses-dev flex bison gperf python3 python3-pip python3-venv cmake ninja-build   ccache libffi-dev libssl-dev dfu-util libusb-1.0-0  
   ```

* Create a directory and Clone the repo

    ```
    mkdir test_build
    cd test_build
    git clone https://github.com/charlie2951/microml.git
    git clone https://github.com/micropython/micropython.git
    cd micropython
    git submodule update --init
    make -C mpy-cross

    ```
    
2. Installing Toolchain <p>

Go back to your created build directory and install toolchain<p>
   **For esp32 port:**
   
   ```
   git clone -b v5.5.1 --recursive https://github.com/espressif/esp-idf.git
   cd esp-idf
   git checkout v5.5.1
   git submodule update --init --recursive
   ./install.sh
   source export.sh # You will need to source export.sh for every new session.
   ```
   <p>
    
   **For `rp2` port:** 
   ```bash
   sudo apt install build-essential git python3 cmake gcc-arm-none-eabi libnewlib-arm-none-eabi
   ```

  3. Build your BOARD <p>
  ```
  cd ports/esp32
  make submodules
  make BOARD=ESP32_GENERIC BOARD_VARIANT=SPIRAM  USER_C_MODULES=../../../../microml/src/  #Find board name in ./boards
  ```
For `rp2` boards<p>
```
cd ports/rp2
make submodules
make BOARD=RPI_PICO  USER_C_MODULES=../../../../microml/src/  #Find board name in ./boards
```
Upon successful build, the `firmware.bin` (also `firmware.uf2` for rp2 port) will be available inside `/ports/esp32/build_board_dir` or `/ports/rp2/build_board_dir`


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
# Architecture: 1 Input -> 8 Hidden Neurons -> 1 Output
mlp_reg = microml.MLPRegressor(1, 16, 1)

# 3. Train the Model
# Parameters: fit(X, y, epochs, learning_rate, momentum)
print("Training MLP Regressor on Sine Wave...")
mlp_reg.fit(X_train, y_train, 5000, 0.001, 0.9)

# 4. Evaluate & Predict
print("\nPredictions vs Actual:")
print("----------------------------")
print("  x   |  Predicted  |   Actual   | Error")
print("----------------------------")

# Test 5 sample points
test_points = [-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2]

for x_test in test_points:
    x_buf = array.array("f", [x_test])

    # Model outputs a single float value directly
    pred = mlp_reg.predict(x_buf)
    actual = math.sin(x_test)
    error = abs(pred - actual)

    print(f"{x_test: .2f} |  {pred: .4f}    |  {actual: .4f}  | {error:.4f}")

```
### Result

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/5ef09b3a-db8c-4b20-a2bb-a5131e490264" />

