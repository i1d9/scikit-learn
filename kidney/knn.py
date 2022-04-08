from calendar import c
import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

data = pd.read_csv("ckd.csv")

print(data)

le = LabelEncoder()


X = data[['age', 'bp', 'sg', 'al']].values
#Converting the features that have string data to numeric data
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

rbc_label_mapping = {
    "normal": 0,
    "abnormal": 1,
    "?": 2
}

pc_label_mapping = {
    "normal": 0,
    "abnormal": 1,
    "?": 2
}

class_label_mapping = {
    "ckd": 0,
    "notckd": 1
}

pcc_label_mapping = {
    "normal": 0,
    "abnormal": 1,
    "?": 2
}

appet_label_mapping = {
    "good": 0,
    "bad" : 1
}

pe_label_mapping = {
    "yes": 0,
    "no": 1
}

ane_label_mapping = {
    "yes": 0,
    "no": 1
}

ba_label_mapping = {
    "present": 0,
    "notpresent": 1
}




#Select the label
y = data[['class']]
y['class'] = y['class'].map(class_label_mapping)
y = np.array(y)



#Create model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights="uniform")

#Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predictions)
print("Predictions:", predictions)
print("Accuracy:", accuracy)

#index_of_instance = 1727
#print("Actual value: ", y[index_of_instance])
#print("Predicted value: ", knn.predict(X)[index_of_instance])

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
jk = confusion_matrix(y_test, predictions)
print(jk)