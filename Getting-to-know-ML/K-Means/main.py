import pandas as pd 
import numpy as np
from sklearn.utils import shuffle

## Load Iris dataset
df = pd.read_csv('/Users/nikhil/Documents/Datasets/Iris_dataset/iris.csv') 
## Store the target vaue
classes = df['Species']  
## Drop the Id and Class values from dat
df = df.drop(['Id','Species'],axis=1) 
## Convert dataframe into list and then into a numpy array
data = df.values.tolist() 
data = np.array(data)
## Shuffle classes and data 
data,classes = shuffle(data,classes) 
## First 135 points are used for training and the rest is used for testing
train_data = data[:135]  
test_data = data[135:]

## K-Means Algorithm
import random
import numpy as np
## Randomly place the centroids of the three clusters 
c1 = [float(np.random.randint(4,8)),float(np.random.randint(1,5)),
      float(np.random.randint(1,7)),float(np.random.randint(0,3))]
c2 = [float(np.random.randint(4,8)),float(np.random.randint(1,5)),
      float(np.random.randint(1,7)),float(np.random.randint(0,3))]
c3 = [float(np.random.randint(4,8)),float(np.random.randint(1,5)),
      float(np.random.randint(1,7)),float(np.random.randint(0,3))]
## Intialize the number of iterations you want to run 
epochs = 1
while(epochs <= 100):
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    for point in train_data:
        ## Find the eucledian distance between all points the centroid
        dis_point_c1 = ((c1[0]-point[0])**2 + (c1[1]-point[1])**2 + 
                        (c1[2]-point[2])**2 + (c1[3]-point[3])**2)**0.5
        dis_point_c2 = ((c2[0]-point[0])**2 + (c2[1]-point[1])**2 + 
                        (c2[2]-point[2])**2 + (c2[3]-point[3])**2)**0.5
        dis_point_c3 = ((c3[0]-point[0])**2 + (c3[1]-point[1])**2 + 
                        (c3[2]-point[2])**2 + (c3[3]-point[3])**2)**0.5
        distances = [dis_point_c1,dis_point_c2,dis_point_c3]
        ## Find the closest centroid to the point and assign the point to that cluster
        pos = distances.index(min(distances))
        if(pos == 0):
            cluster_1.append(point)
        elif(pos == 1):
            cluster_2.append(point)
        else:
            cluster_3.append(point)
    ## Store the centroid values to calculate new centroid values 
    prev_c1 = c1
    prev_c2 = c2
    prev_c3 = c3
    cluster_1 = np.array(cluster_1)
    cluster_2 = np.array(cluster_2)
    cluster_3 = np.array(cluster_3)
    ## Find mean of all points within a cluster and make it as the centroid 
    if(len(cluster_1) != 0):
        c1 = [sum(cluster_1[:,0])/float(len(cluster_1)),
              sum(cluster_1[:,1])/float(len(cluster_1)),
              sum(cluster_1[:,2])/float(len(cluster_1)),
              sum(cluster_1[:,3])/float(len(cluster_1))]
    if(len(cluster_2) != 0):
        c2 = [sum(cluster_2[:,0])/float(len(cluster_2)),
              sum(cluster_2[:,1])/float(len(cluster_2)),
              sum(cluster_2[:,2])/float(len(cluster_2)),
              sum(cluster_2[:,3])/float(len(cluster_2))]
    if(len(cluster_3) != 0):
        c3 = [sum(cluster_3[:,0])/float(len(cluster_3)),
              sum(cluster_3[:,1])/float(len(cluster_3)),
              sum(cluster_3[:,2])/float(len(cluster_3)),
              sum(cluster_3[:,3])/float(len(cluster_3))]
    ## If centroid values hasn't changed, algorithm has convereged 
    if(prev_c1 == c1 and prev_c2 == c2 and prev_c3 == c3):
        print("Converged")
        break
    print(epochs)
    epochs += 1
    
    pred = []
for point in test_data:
    ## Find distance between test data point and centroids
    dis_point_c1 = ((c1[0]-point[0])**2 + (c1[1]-point[1])**2 + 
                    (c1[2]-point[2])**2 + (c1[3]-point[3])**2)**0.5
    dis_point_c2 = ((c2[0]-point[0])**2 + (c2[1]-point[1])**2 + 
                    (c2[2]-point[2])**2 + (c2[3]-point[3])**2)**0.5
    dis_point_c3 = ((c3[0]-point[0])**2 + (c3[1]-point[1])**2 + 
                    (c3[2]-point[2])**2 + (c3[3]-point[3])**2)**0.5
    ## Find the cluster to which the point is closest to and append 
    ## it to pred
    distances = [dis_point_c1,dis_point_c2,dis_point_c3]
    pos = distances.index(min(distances))
    pred.append(pos)
    ## Print the predictions 
    print(pred)
    
    from sklearn.cluster import KMeans 

clf = KMeans(n_clusters = 3)
clf.fit(train_data)
pred = clf.predict(test_data)
