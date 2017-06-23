#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 07:47:33 2017

@author: r3gz3n
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

def rmse(pred, y_test):
    return np.sqrt(np.sum((pred - y_test) ** 2) / len(pred))

train_data = pd.read_csv("train_new1.csv")
test_data = pd.read_csv("test_new1.csv")

trainy_data = train_data['SalePrice']
trainx_data = train_data[train_data.columns.difference(['SalePrice', 'Id'])]
testx_data = test_data[test_data.columns.difference(['Id'])]

X = trainx_data.as_matrix()
y = trainy_data.as_matrix()
X_test = testx_data.as_matrix()

# Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(X, y)
pred_lr = lr.predict(X_test)


# Neural Network

nn = MLPRegressor(hidden_layer_sizes=(80,), solver='lbfgs', max_iter=2000)
nn.fit(X, y)
pred_nn = nn.predict(X_test)


pred = (0.3*pred_nn + 0.7*pred_lr)

d = {'Id':test_data['Id'], 'SalePrice':pred}
df = pd.DataFrame(data=d)
df.to_csv('submission_combined.csv')