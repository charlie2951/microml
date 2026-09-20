#Demonstration of csvread and csvwrite
import microml
from microml.utils import csvread, csvwrite
from ulab import numpy as np
data=csvread("random_data.csv") # reading from csv file (only comma separated)
print(data)
x=np.linspace(1,20,20)
x=x.reshape((4,5))
print(x)
csvwrite("xdata.csv",x)#write to csv file
