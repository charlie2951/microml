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
