#!/usr/bin/python
import sys
from time import time
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Starting K-nearest neighbors.........")
clf1 = KNeighborsClassifier(n_neighbors= 1)

t1_k = time()
clf1.fit(features_train,labels_train)
print("Time to train : ", round(time()-t1_k,3))

t2_k = time()
pred1 = clf1.predict(features_test)
print("Time to predict : ", round(time()-t2_k,3))

print("Accuracy of K-nearest neighbors : ",accuracy_score(labels_test,pred1)*100, "%" )

print("Starting AdaBoost.........")
clf2 = AdaBoostClassifier(n_estimators=11)

t1_Ada = time()
clf2.fit(features_train,labels_train)
print("Time to train : ", round(time()-t1_Ada,3))

t2_Ada = time()
pred2 = clf2.predict(features_test)
print("Time to predict : ", round(time()-t2_Ada,3))

print("Accuracy of AdaBoost : ",round(accuracy_score(labels_test,pred2)*100,3), "%" )

print("Starting Random Forest.........")
clf3 = RandomForestClassifier(max_depth=3,n_estimators=100)

t1_rf = time()
clf3.fit(features_train,labels_train)
print("Time to train : ", round(time()-t1_rf,3))

t2_rf = time()
pred3 = clf3.predict(features_test)
print("Time to predict : ", round(time()-t2_rf,3))

print("Accuracy of Random Forest : ",round(accuracy_score(labels_test,pred3)*100,3), "%" )

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
