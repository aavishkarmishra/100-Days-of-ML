#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

from sklearn.metrics import accuracy_score
from sklearn import tree
from email_preprocess import preprocess
import sys
from time import time
sys.path.append("/media/aavishkar/Data/100-Days-of-ML/tools/")


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###


clf = tree.DecisionTreeClassifier(min_samples_split=40)

t1 = time()
clf = clf.fit(features_train, labels_train)
print("Time to train : ", round(time() - t1, 3))

t2 = time()
pred = clf.predict(features_test)
print("Time to predict : ", round(time() - t2, 3))

acc = accuracy_score(labels_test, pred)

print("Accuracy : ", acc*100, "%")

#########################################################
