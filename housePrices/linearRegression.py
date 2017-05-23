#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:12:11 2017

@author: r3gz3n
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


train_data = pd.read_csv("train_new1.csv")
test_data = pd.read_csv("test_new1.csv")

trainy_data = train_data['SalePrice']
trainx_data = train_data[train_data.columns.difference(['SalePrice', 'Id'])]
testx_data = test_data[test_data.columns.difference(['Id'])]

X = trainx_data.as_matrix()
y = trainy_data.as_matrix()
X_test = testx_data.as_matrix()


lr = LinearRegression(normalize=True)
lr.fit(X, y)
pred = lr.predict(X_test)
d = {'Id':test_data['Id'], 'SalePrice':pred}
df = pd.DataFrame(data=d)
df.to_csv('submission_linearRegression.csv')
