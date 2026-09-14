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
* **Embedded Resource Friendly:**
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
1. Build System Setup<p>
Using micropython.mk (Make-based ports: ESP32, STM32, UNIX)<p>
Add both microml.c and mlp.c to SRC_USERMOD_C:<p>

```cmake
USERMOD_DIR_MICROML := $(USERMOD_DIR)

SRC_USERMOD_C += $(USERMOD_DIR_MICROML)/microml.c
SRC_USERMOD_C += $(USERMOD_DIR_MICROML)/mlp.c

INC += -I$(USERMOD_DIR_MICROML)
```
