# -*- coding: utf-8 -*-
"""

Principal Component Analysis With KNeghbors Classifier On Wine Segment  Dataset
@author: Rahul
"""

 #Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Getting  the dataset
dataset = pd.read_csv('wine_segment.csv')
x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# Applying PCA
princom = PCA(n_components = 2)
x_train = princom.fit_transform(x_train)
x_test = princom.transform(x_test)
explained_variance = princom.explained_variance_ratio_

# Fitting KNN Classification mehtod to the training data 
knighborclassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knighborclassifier.fit(x_train, y_train)

# Predicting  results
y_pred =knighborclassifier.predict(x_test)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)
print(cm) 
print(accuracy)
print(classification_report(y_test, y_pred))