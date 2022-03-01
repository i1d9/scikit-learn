from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

#Split it in features and labels
X = iris.data
y = iris.target

print(X.shape)
print(y.shape)

#Hours of study vs good/bad grades

