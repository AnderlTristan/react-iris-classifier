import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn import datasets

# SciKit features some small toy datasets that can be used for learning ML or for
# benchmarking algorithms that are used for 'real world' applications
dataset = datasets.load_iris()

# The iris dataset will have the following information
# 150 instances
# 4 numeric attributes:
# sepal length (cm)
# sepal width (cm)
# petal length (cm)
# petal width (cm)
# class:
# Iris-Setosa, Iris-Versicolour, Iris-Virginica

# array of shape n_samples * n_features
# key: data
X = dataset.data
# numpy array of length n_samples
# key: target
y = dataset.target

# train_test_split(*arrays, **options)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)

classifier = DecisionTreeClassifier()
# Builds decision tree classifier from the training set (X, y)
classifier.fit(X_train, y_train)
# Predicts the class value for X
prediction = classifier.predict(X_test)

# Evaluates accuracy of a classification
print("Confusion Matrix:")
# confusion_matrix(y_true, y_pred)
# y_true = ground truth target values
# y_pred = estimated targets returned by classifier
print(confusion_matrix(y_test, prediction))

# persisting object into one file
joblib.dump(classifier, 'classifier.joblib')