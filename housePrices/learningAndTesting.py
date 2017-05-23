#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 18:42:17 2017

@author: r3gz3n
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("train_new1.csv")

trainy_data = train_data['SalePrice']
trainx_data = train_data[train_data.columns.difference(['SalePrice', 'Id'])]

X = trainx_data.as_matrix()
y = trainy_data.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

"""
# scaling
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
"""


# linear regression
lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
error_lr = np.sqrt(np.sum((pred - y_test) ** 2) / len(pred))

# neural network

nn = MLPRegressor(hidden_layer_sizes=(80,), solver='lbfgs', max_iter=2000)
nn.fit(X_train, y_train)
pred = nn.predict(X_test)
error_nn = np.sqrt(np.sum((pred - y_test) ** 2) / len(pred))
print(error_nn, error_lr)

"""
# decision tree
regr_1 = DecisionTreeRegressor(max_depth=10)
regr_1.fit(X_train, y_train)
pred_1 = regr_1.predict(X_test)
error_regr_1 = np.sqrt(np.sum((pred_1 - y_test) ** 2) / len(pred_1))

feat_1 = []
idx = 0
for label in list(trainx_data):
    feat_1.append((regr_1.feature_importances_[idx], label))
    idx += 1
feat_1.sort(reverse=True)
labels = []
for i in range(76):
    if feat_1[i][0] < 0.00001:
        break
    labels.append(feat_1[i][1])
print(labels)
"""