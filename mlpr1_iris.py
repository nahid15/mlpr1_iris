# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:49:14 2018

@author: ASHIK
"""

from sklearn import naive_bayes,svm,neighbors,ensemble
import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
df = pd.read_csv('iris.data.csv') #Dataset taken from UCI ML repository
data = np.array(df) #converted into numpy array

#cross validation will be done manually, 120 samples for training and 30 for testing
np.random.shuffle(data) #shuffled data for better evaluation
X_train = np.array(data[0:120,0:4]) #used the first 120 rows and 4 colums to train the model
y_train = np.array(data[0:120,4]) #120 lebels
X_test = np.array(data[120:150,0:4]) #30 rows and 4 colums to test on the model
y_test = np.array(data[120:150,4]) #30 lebels to evaluate the performance of the model

#scikit learn ML library for cross validation
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) 

#Various Machine learning algorithms for classification
#clf = naive_bayes.GaussianNB()
clf = svm.SVC(C=10,kernel='poly')
#clf = neighbors.KNeighborsClassifier(n_neighbors = 5,algorithm='auto')
#clf = ensemble.AdaBoostClassifier(n_estimators=80,learning_rate=1.3)
#clf = ensemble.RandomForestClassifier()

clf.fit(X_train,y_train) #training of the model
predict = clf.predict(X_test) #prediction based on the training
accuracy = clf.score(X_test,y_test) #accuracy calculation by using test samples.
                                     #Accuracy may vary between 93%-100% as the data is shuffled

print('The accuracy is:',accuracy*100,'%')
print('Actual outputs:',y_test)
print('Predicted outputs',predict)
