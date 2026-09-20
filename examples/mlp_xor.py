#XOR training demo
import microml
from ulab import numpy as np
import mlp_model as m
print("MLP XOR Prediction---->")
X_train=np.array([[0,0],[0,1],[1,0],[1,1]])
y_train=np.array([0,1,1,0])
mlp = microml.MLP([2,4,4,2], False)
mlp.fit(X_train, y_train, 3000, 0.01, 0.9)
y_pred = []
for sample in X_train:
    label = mlp.predict(sample)
    y_pred.append(label)
print(y_pred)
y_p=np.array(y_pred) #convert list to array
#confusion matrix
from microml.metrics import confusion_matrix, classification_report 
confusion_matrix(y_train, y_p,title='XOR Prediction')
classification_report(y_train,y_p, digits=4)


