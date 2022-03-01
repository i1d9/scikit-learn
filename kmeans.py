from statistics import mode
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

bc = load_breast_cancer()

X = scale(bc.data)
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KMeans(n_clusters=2, random_state=0)

#We dont pass y_train because in clustering y is the output not the input
model.fit(X_train)

predictions = model.predict(X_test)
lables = model.labels_

print("labels", lables)
print("predictions", predictions)
print("accuracy", accuracy_score(y_test, predictions))
print("actual", y_test)

print(pd.crosstab(y_train, lables))