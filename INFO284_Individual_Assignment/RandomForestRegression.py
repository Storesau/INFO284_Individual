#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Getting the dataset
dataset = pd.read_csv('Flaveria.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 1] = labelencoder.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(n_values = [3, 6])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Decision Tree Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting result based on test set
y_pred = regressor.predict(X_test)
#y_pred2 = regressor.predict(5.4)

# Scoring model
regressor.score(X_test, y_test)

# Applying k-fold cross validations
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
