#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from email_preprocess import preprocess
import sys
from time import time
sys.path.append("/media/aavishkar/Data/100-Days-of-ML/tools/")


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ##

nb_clf = GaussianNB()
# Training the classifier with the training data
t = time()
nb_clf.fit(features_train, labels_train)
print("Time to train: ", round(time() - t, 3))

t1 = time()
pred = nb_clf.predict(features_test)
print("Time to predict: ", round(time() - t1, 3))

acc = accuracy_score(pred, labels_test)
print("Accuracy=", acc*100, "%")

#########################################################
