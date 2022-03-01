from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()

#Features
X = boston.data
#Lables
y = boston.target

print(f"X\n{X}\n{X.shape}\n")


#Algorithm
l_reg = linear_model.LinearRegression()

plt.scatter(X.T[5], y)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Train the model
model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions: ", predictions)
"""
R-squared is a goodness-of-fit measure for linear regression models. 
This statistic indicates the percentage of the variance in the dependent variable that the independent variables explain collectively
"""
print("R^2 value: ", l_reg.score(X, y))
print("coedd: ", l_reg.coef_)
print("intercept: ", l_reg.intercept_)