#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py <input> <output>
"""
"""import sys

content = ''
outsize = 0
with open('D:/100-Days-of-ML/final_project/final_project_dataset.pkl', 'rb') as infile:
  content = infile.read()
with open('D:/100-Days-of-ML/final_project/final_project_dataset_new.pkl', 'wb') as output:
  for line in content.splitlines():
    outsize += len(line) + 1
    output.write(line + b'\n')

print("Done. Saved %s bytes." % (len(content)-outsize))"""

import pickle

enron_data = pickle.load(open('D:/100-Days-of-ML/final_project/final_project_dataset_new.pkl', 'rb'))

print("Loading done.")

"""count = 0
for i in enron_data:
    if enron_data[i]['poi']:
        count += 1
print (count)"""

print(enron_data["SKILLING JEFFREY K"])

