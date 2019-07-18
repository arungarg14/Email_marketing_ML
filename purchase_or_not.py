#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.drop('User ID', axis=1, inplace=True)

#Initialize the dependent and independent variable
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3:4].values

#Splitting dataset into traing and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#for better visualize
X_test_real = X_test  
X_test_real = X_test_real.astype(str)

#convert 0th index of X_train and X_test into Categorial values
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_train[:,0] = labelencoder_X.fit_transform(X_train[:,0])
X_test[:,0] = labelencoder_X.transform(X_test[:,0])

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting data into KNeighbors 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix (For result)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)

#calculate accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)