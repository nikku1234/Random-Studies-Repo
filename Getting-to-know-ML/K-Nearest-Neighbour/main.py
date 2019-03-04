import pandas as pd 

## Load Iris dataset
df = pd.read_csv('/Users/nikihl/Documents/Datasets/Iris_dataset/Iris.csv')
## Retrieve the target values and drop the Id along with it
target = df['Species']
df = df.drop(['Species','Id'],axis=1)
## Drop the two features we won't be using from the dataframe 
df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)

import matplotlib.pyplot as plt 

points_1 = df[0:50].values.tolist()
points_2 = df[50:100].values.tolist()
points_3 = df[100:].values.tolist()

points_1 = np.array(points_1)
points_2 = np.array(points_2)
points_3 = np.array(points_3)

## Plot the data points to visualize 
plt.figure(figsize=(16,9))
plt.scatter(points_1[:,0],points_1[:,1],color='red',label='Setosa')
plt.scatter(points_2[:,0],points_2[:,1],color='black',label='Versicolor')
plt.scatter(points_3[:,0],points_3[:,1],color='green',label='Virginica')
plt.legend()
plt.show()


import numpy as np 
from sklearn.cross_validation import train_test_split

## Retrieve features
X = df.values.tolist()
Y = []
## Convert classes in Strings to Integers 
for val in target:
    if(val == 'Iris-setosa'):
        Y.append(0)
    elif(val == 'Iris-virginica'):
        Y.append(2)
    else:
        Y.append(1)
## Make them as numpy array         
X = np.array(X)
Y = np.array(Y)
## Shuffle and split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.9)
## Make them as numpy arrays 
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

## We take a range of values for K(1 to 20) and find the accuracy 
## so that we can visualize how accuracy changes based on value of K
accuracy = []
for n in range(1,21):
    clf = KNeighborsClassifier(n_neighbors = n)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    accuracy.append(accuracy_score(y_test,y_pred))
## Plotting the accuracies for different values of K
plt.figure(figsize=(16,9))
plt.plot(range(1,21),accuracy)

## K Nearest Neighbors 

y_pred_knn = []
## Iterate through each value in test data 
for val in x_test:
    euc_dis = []
    ## Finding eucledian distance for all points in training data 
    for point in x_train:
        euc_dis.append(((val[0]-point[0])**2+(val[1]-point[1])**2)**0.5)
    temp_target = y_train.tolist()
    ## Use bubble sort to sort the euclidean distances 
    for i in range(len(euc_dis)):
        for j in range(0,len(euc_dis)-i-1):
            if(euc_dis[j+1] < euc_dis[j]):
                euc_dis[j], euc_dis[j+1] = euc_dis[j+1], euc_dis[j]
                ## Sort the classes along with the eucledian distances 
                ## to maintain relevancy 
                temp_target[j], temp_target[j+1] = temp_target[j+1], temp_target[j] 
    ## Finding majority among the neighbours 
    vote = [0,0,0]
    ## We are using only the first three entries (K = 3)
    for i in range(3):
        vote[temp_target[i]] += 1
    y_pred_knn.append(vote.index(max(vote)))
## Print the accuracy score     
print('Accuracy:',accuracy_score(y_test,y_pred_knn)) 



