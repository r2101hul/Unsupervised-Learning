# -*- coding: utf-8 -*-
"""
K Means Clustring 

@author: Rahul
"""
#Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#Find the Number of Clusters
wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()   

#Kmeans Clustring
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)

#Prediction of clusters
y_pred=kmeans.fit_predict(X)

#Visualization of clusters
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=200,c='red')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='green')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='blue')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c='cyan')
plt.scatter(X[y_pred==4,0],X[y_pred==4,1],s=100,c='magenta')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

