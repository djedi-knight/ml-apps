#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:34:35 2018

@author: sdaichendt
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap (done automatically by most algorithms)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(X_test)

# Building the Optimal Model using Backward Elimination
import statsmodels.formula.api as sm
# Prepend column of ones to X to account for b constant
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# Initialize optimal model
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Create OLS regressor and fit optimal model 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
# Initialize optimal model (minus predictor)
X_opt = X[:, [0, 1, 3, 4, 5]]
# Create OLS regressor and fit optimal model 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
# Initialize optimal model (minus predictor)
X_opt = X[:, [0, 3, 4, 5]]
# Create OLS regressor and fit optimal model 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
# Initialize optimal model (minus predictor)
X_opt = X[:, [0, 3, 5]]
# Create OLS regressor and fit optimal model 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
# Initialize optimal model (minus predictor)
X_opt = X[:, [0, 3]]
# Create OLS regressor and fit optimal model 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()