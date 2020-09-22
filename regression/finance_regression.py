#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    
'''
import sys

content = ''
outsize = 0
with open('../final_project/final_project_dataset_modified.pkl', 'rb') as infile:
  content = infile.read()
with open('../final_project/final_project_dataset_modified_new.pkl', 'wb') as output:
  for line in content.splitlines():
    outsize += len(line) + 1
    output.write(line + b'\n')

print("Done. Saved %s bytes." % (len(content)-outsize))'''

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "rb") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from time import time
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
t = time()
reg.fit(feature_train, target_train)
print ("Time to train: ", round(time() - t, 3))
t = time()
pred = reg.predict(feature_test)
print ("Time to predict: ", round(time() - t, 3))
acc = reg.score(feature_test, target_test)
print("Accuracy: ",acc*100,"%")
coeff = reg.coef_
inter = reg.intercept_
print ("Coefficient: ",coeff)
print ("Intercept: " , inter)







### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
