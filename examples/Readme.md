# Examples
## Regression problem: Prediction of Sine wave (see [code](https://github.com/charlie2951/microml/blob/main/examples/mlp_sinewave.py))

1. Generate evenly spaced x points in range [-pi, pi] for training <p>
   `X_train = np.linspace(-math.pi, math.pi, 50)`
2. Compute y values using vectorized numpy sine function<p>
   `y_train = np.sin(X_train)`
3. Instantiate MLPRegressor <p>
    ```
    #Architecture: 1 Input -> 16 Hidden Neurons -> 8 Hidden Neurons -> 1 Output
    mlp_reg = microml.MLP([1, 16, 8, 1], True)
    ```
4. Train the Model
    ```
    print("Training MLP Regressor on Sine Wave...")
    mlp_reg.fit(X_train, y_train, 1000, 0.001, 0.9)
    ```
5. Evaluate & Predict   
    ```
    print("\nPredictions vs Actual:")
    print("---------------------------------------")
    print("  x    |  Predicted  |   Actual   | Error")
    print("---------------------------------------")

    #Test 5 sample points
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
    
    ```
6. Save the model for future uses
      
      ```
      mlp_reg.save("mlp_sine.bin")
      print("Model saved!")
      ```
7. Load the saved model
     
      ```
      #Load model weights into an existing instance
      saved_model = microml.MLP([2, 1])  # dummy model, Dimensions will be updated on load
      saved_model.load("mlp_sine.bin")
     
      ```
    ### Console output
<img width="422" height="306" alt="image" src="https://github.com/user-attachments/assets/7eeb7b44-4cc8-4ced-88f1-db71c27395b7" />

## Classification problem: IRIS Flower classification (using MLP)

```
# IRIS flower classification using MLP model
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
#print report and confusion matrix
labels = ["Setosa", "Versicolor", "Virginica"]
microml.confusion_matrix(y_true, y_p, num_classes=3, labels=labels, title="IRIS flower classification")
microml.classification_report(y_true, y_p, labels=labels, digits=4)
```
