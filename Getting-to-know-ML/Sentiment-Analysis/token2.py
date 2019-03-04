import numpy as np 
from sklearn.model_selection import train_test_split

Y = []
for val in y:
    if(val == 0):
        Y.append([1,0])
    else:
        Y.append([0,1])
Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.8)
