from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()

#Features
X = boston.data
#Lables
Y = boston.target

print(f"X\n{X}\n{X.shape}\n")
