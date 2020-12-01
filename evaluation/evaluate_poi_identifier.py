#!/usr/bin/python
"""
    Starter code for the evaluation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from feature_format import featureFormat, targetFeatureSplit
import pickle
import sys
sys.path.append("../tools/")

data_dict = pickle.load(
    open("../final_project/final_project_dataset.pkl", "rb"))

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# Time to split the data!
X_train, X_test, Y_train, Y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# Creating the classifier, fitting the data and predicting from the test set
clf = DecisionTreeClassifier()
t = time()
clf.fit(X_train, Y_train)
print("Time to train: ", round(time()-t, 3))
t = time()
pred = clf.predict(X_test)
print("Time to Predict: ", round(time()-t, 3))

# Print the accuracy score for the decision tree
acc = accuracy_score(pred, Y_test)
print("Accuracy is : ", acc)

b = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
a = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
precision = precision_score(a, b)
recall = recall_score(a, b)
print("Precision score: ", precision)
print("Recall score: ", recall)
