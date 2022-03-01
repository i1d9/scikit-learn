import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("car.data")
print(data.head())

#Select some features
X = data[['buying', 'maint', 'safety']]

#Select the label
y = data[['class']]

print(X, y)