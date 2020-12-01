#!/usr/bin/python

from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from feature_format import featureFormat, targetFeatureSplit
import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


# read in data dictionary, convert to numpy array
data_dict = pickle.load(StrToBytes(
    open("../final_project/final_project_dataset.pkl", "r")))
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


# your code below

X = []
Y = []
i = 0

# Initial plot to visualize outliers and gather the features and labels
for sample in data:
    salary = sample[0]
    bonus = sample[1]
    if(i == 67):
        i += 1
        continue
    X.insert(i, salary)
    Y.insert(i, bonus)
    i += 1
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
