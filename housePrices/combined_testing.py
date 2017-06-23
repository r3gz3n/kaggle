#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 07:33:37 2017

@author: r3gz3n
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def rmse(pred, y_test):
    return np.sqrt(np.sum((pred - y_test) ** 2) / len(pred))

train_data = pd.read_csv("train_new1.csv")

trainy_data = train_data['SalePrice']
trainx_data = train_data[train_data.columns.difference(['SalePrice', 'Id'])]

X = trainx_data.as_matrix()
y = trainy_data.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)


# Neural Network

nn = MLPRegressor(hidden_layer_sizes=(80,), solver='lbfgs', max_iter=2000)
nn.fit(X_train, y_train)
pred_nn = nn.predict(X_test)


pred = (0.3*pred_nn + 0.7*pred_lr)


print("Linear Regression: ", rmse(pred_lr, y_test))
print("Neural Networks: ", rmse(pred_nn, y_test))
print("Combined: ", rmse(pred, y_test))