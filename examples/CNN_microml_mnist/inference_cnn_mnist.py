#MNIST Digit prediction using trained model in Tensorflow
import microml
from ulab import numpy as np
from microml.utils import csvread
# Instantiate CNN with exact parameters: 
# (Height, Width, Channels), out_channels, kernel_size, pool_size, [dense_hidden, dense_output]
model = microml.CNN((28, 28, 1), 4, 3, 2, [16, 10])

# Load exported binary file
model.load("cnn_mnist.bin")
#read csv file
digit_data=csvread("digit2.csv") # try with different file for prediction

actual_digit=digit_data[0][0] #actual label
digit_data=digit_data[0][1:]/255.0 #load rest starting from indx 1 and normalize
# Run prediction
predicted_digit = model.predict(digit_data)
print(f"Predicted digit: {predicted_digit} , Actual digit: {int(actual_digit)}")

