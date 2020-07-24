#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("D:/100-Days-of-ML/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn import svm



clf = svm.SVC(kernel='rbf',C= 10000.0)

t1 = time()
clf.fit(features_train, labels_train)
print ("Time to train: ", round(time() - t1, 3))

t2 = time()
pred=clf.predict(features_test)
print ("Time to predict: ", round(time() - t2, 3))

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("Accuracy=",acc*100,"%")


#########################################################


