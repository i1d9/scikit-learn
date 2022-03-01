#SVM is effective for high dimensional spaces
import imp
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

iris = datasets.load_iris()

#Split it in features and labels
X = iris.data
y = iris.target


classes = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]

print(X.shape)
print(y.shape)


#Split and train using only 20% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_train.shape)


model = svm.SVC()
model.fit(X_train, y_train)

print(model)