![GitHub commit activity](https://img.shields.io/github/commit-activity/t/charlie2951/microml?label=Total%20commit) ![GitHub Repo stars](https://img.shields.io/github/stars/charlie2951/microml?style=flat) ![Build Status](https://github.com/charlie2951/microml/actions/workflows/build-release.yml/badge.svg)


# MicroML: Lightweight Machine Learning Module for MicroPython

`MicroML` is a lightweight, zero-dependency module built for MicroPython that enables on-device training and inference directly on microcontrollers (e.g., ESP32, RP2040, STM32, nRF52).

It features classical ML algorithms as well as an optimized **Multi-Layer Perceptron (MLP)** engine supporting non-linear regression using Mean Squared Error (MSE) loss and Stochastic Gradient Descent (SGD) with momentum. To use it, you need to use the custom firmware where these API are written in C, compiled as external C modules, and frozen inside the firmware, thus providing optimum performance

---

> [!NOTE]
> From releases V2.0.0, `ulab` (NumPy-style package for Micropython) is also included with firmware to make array operations easier. For more details about **`ulab`**, see [here](https://micropython-ulab.readthedocs.io/en/latest) <p>
> New Features available from V2.0.2:<p>
> * `csvwrite()` and `csvread()` for working with csv files<p>
> * `classification_report()` and `confusion_matrix()` print<p>
> * `StandardScaler` and `MinMaxScaler` for preprocessing <p>
> * `model.save()` and `model.load()` to save a trained MLP model and to load it.<p>

## Features

* **Multi-Layer Perceptron (MLP):**
  * Flexible multi-layer architecture (Input $\to$ Hidden(s) $\to$ Output).
  * ReLU activation for hidden layers and Linear activation for continuous output regression.
  * Backpropagation engine powered by SGD with momentum acceleration.
  * Real-time training loss logging outputted directly to the MicroPython REPL using `mp_printf`.
* **Classical ML Algorithms:**
  * **K-Nearest Neighbors (KNN)** with confidence probabilities.
  * **Decision Tree** with Gini impurity splitting and binary `save`/`load` file serialization.
  * **Support Vector Machine (SVM)** supporting both Linear and Radial Basis Function (RBF) kernels.
* **Embedded Resource-Friendly:**
  * Support NumPy-like array operations using `ulab` (available from v2.0.0 firmware)
  * Zero dynamic memory fragmentation via MicroPython GC heap routines (`m_new`, `m_free`).
  * Direct buffer protocol execution on native Python `array.array('f')` and `array.array('i')` data types without extra memory copying.
  * Also supports ulab's `ndarray`

---

## Repository Structure

```text
microml/
├── examples/       # MicroPython bindings & C-module wrappers
├── firmware/       # pre-built firmware for some port(not updated, download from release)
├── manifest.py     # Manifest file for py module compilation
├── build.sh        # custom build script for Ubuntu22.04 Python 3.10
├── src/           # C-module and python wrappers
       ├──microml.c       # MicroPython bindings & C-module wrappers
       ├──microml.py       # MicroPython python module bindings 
       ├── microml.h       # Headers for KNN, DecisionTree, and SVM implementations
       ├── mlp.c           # MLP Forward/Backward propagation engine
       ├── mlp.h           # MLP model definitions and memory interfaces
       ├── micropython.cmake  # CMake configuration
       ├── microml  # python module binding
           ├── __init__.py
           ├── metrics.py
           ├── utils.py
           ├── preprocessing.py
```
## Pre-compiled firmware
Pre-built firmware for some ports is available on [**release**](https://github.com/charlie2951/microml/releases). Note: Not all firmwares are tested on hardware.

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
    git clone https://github.com/v923z/micropython-ulab.git
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
  make BOARD=ESP32_GENERIC BOARD_VARIANT=SPIRAM  USER_C_MODULES=/path/to/microml/  #Find board name in ./boards
  ```
For `rp2` boards<p>
```
cd ports/rp2
make clean
make submodules
make BOARD=RPI_PICO  USER_C_MODULES=/path/to/microml/ FROZEN_MANIFEST="/path/to/microml/manifest.py" #Find board name in ./boards
```
Upon successful build, the `firmware.bin` (also `firmware.uf2` for rp2 port) will be available inside `/ports/esp32/build_board_dir` or `/ports/rp2/build_board_dir`

For **`Unix`** port (experimenting new features!)<p>
```
cd ports/unix
make clean
make submodules
make USER_C_MODULES="/path/to/microml/src /path/to/micropython-ulab/code/ FROZEN_MANIFEST="/path/to/microml/manifest.py"  
```
Now go to `/build-standard` and run `./micropython` in your shell.

## Python API Reference (see examples for working demo)
1. Multi-Layer Perceptron Regressor `microml.MLP` Constructor: <p>
- `microml.MLP([input_dim, hidden_dim1,hidden_dim2..., output_dim], True/False)`: Allocates model weights and momentum buffers on the MicroPython heap with Xavier uniform random initialization. If True-> Regression, False-> classification
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

## IRIS Flower classification using MLP 
```python
import array
import microml
from ulab import numpy as np
# 1. Multi-Class Dataset (6 training samples, 4 features)
X_train = np.array( [
    [5.1, 3.5, 1.4, 0.2], # Setosa (0)
    [4.9, 3.0, 1.4, 0.2], # Setosa (0)
    [7.0, 3.2, 4.7, 1.4], # Versicolor (1)
    [6.4, 3.2, 4.5, 1.5], # Versicolor (1)
    [6.3, 3.3, 6.0, 2.5], # Virginica (2)
    [5.8, 2.7, 5.1, 1.9]  # Virginica (2)
])
y_train = np.array([0, 0, 1, 1, 2, 2])

# 2. Train MLP Model (4 inputs, 8 hidden, 3 outputs)
nn = microml.MLP([4, 8, 8, 3], False)
nn.fit(X_train, y_train, 5000, 0.005, 0.9)

# 3. Test Set (Ground Truth vs Model Predictions)
X_test = np.array([
    [5.0, 3.4, 1.5, 0.2], # True: 0
    [4.8, 3.1, 1.6, 0.2], # True: 0
    [6.2, 2.9, 4.3, 1.3], # True: 1
    [5.9, 3.0, 4.2, 1.5], # True: 1
    [6.5, 3.0, 5.2, 2.0], # True: 2
    [5.7, 2.8, 4.5, 1.3]  # True: 1 (Edge case)
])
y_true = np.array([0, 0, 1, 1, 2, 2])

# Collect Model Predictions
y_pred = []
for sample in X_test:
    label= nn.predict(sample)
    y_pred.append(label)

y_p=np.array(y_pred)

labels = ["Setosa", "Versicolor", "Virginica"]
#microml.classification_report(y_true, y_pred, labels=labels)
microml.confusion_matrix(y_true, y_p, num_classes=3, labels=labels, title="IRIS flower classification")
microml.classification_report(y_true, y_p, labels=labels, digits=4)

```
### Result

```
MPY: soft reboot
Epoch 500/5000 - Loss: 0.0003 - Accuracy: 100.0%
Epoch 1000/5000 - Loss: 0.0001 - Accuracy: 100.0%
Epoch 1500/5000 - Loss: 0.0001 - Accuracy: 100.0%
Epoch 2000/5000 - Loss: 0.0001 - Accuracy: 100.0%
Epoch 2500/5000 - Loss: 0.0000 - Accuracy: 100.0%
Epoch 3000/5000 - Loss: 0.0000 - Accuracy: 100.0%
Epoch 3500/5000 - Loss: 0.0000 - Accuracy: 100.0%
Epoch 4000/5000 - Loss: 0.0000 - Accuracy: 100.0%
Epoch 4500/5000 - Loss: 0.0000 - Accuracy: 100.0%
Epoch 5000/5000 - Loss: 0.0000 - Accuracy: 100.0%

--- IRIS flower classification ---
Actual \ Pred   |     Setosa Versicolor  Virginica
---------------------------------------------------
Setosa          |          2          0          0
Versicolor      |          0          2          0
Virginica       |          0          1          1
---------------------------------------------------
Accuracy: 83.33% (5/6)

==========================================================
                  CLASSIFICATION REPORT
==========================================================
Class           Precision     Recall   F1-Score    Support
----------------------------------------------------------
Setosa             1.0000     1.0000     1.0000          2
Versicolor         0.6667     1.0000     0.8000          2
Virginica          1.0000     0.5000     0.6667          2
----------------------------------------------------------
Accuracy                                 0.8333          6
Macro Avg          0.8889     0.8333     0.8222          6
Weighted Avg       0.8889     0.8333     0.8222          6
==========================================================

```


