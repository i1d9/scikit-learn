from calendar import c
import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("car.data")
#print(data.head())

#Select some features
X = data[['buying', 'maint', 'safety']].values

#Select the label
y = data[['class']]
print(X, y)

#Converting the features that have string data to numeric data
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

print(X)


#Converting the label to numeric data
label_mapping = {
    "unacc":0,
    "acc":1,
    "good":2,
    "vgood":3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)
print(y)