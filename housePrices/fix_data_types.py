#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:53:28 2017

@author: r3gz3n
"""

import pandas as pd
import numpy as np

train_datax = pd.read_csv('train.csv')
test_datax = pd.read_csv('test.csv')

labels = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', 'WoodDeckSF', '1stFlrSF', '2ndFlrSF', 'LotArea', 'GarageArea', 'MSSubClass', 'LotFrontage', 'GarageType', 'BsmtFinType1', 'GarageYrBlt', 'YearBuilt', 'GarageCars', 'OverallCond', 'OpenPorchSF', 'Neighborhood', 'YearRemodAdd', 'BsmtUnfSF', 'KitchenAbvGr', 'BsmtQual', 'BedroomAbvGr', 'Exterior2nd', 'MasVnrArea', 'GarageQual', 'KitchenQual', 'Condition1', 'RoofStyle', 'MSZoning', 'ExterCond', 'CentralAir', 'LandContour', 'Functional', 'BsmtFullBath', 'FullBath', 'TotRmsAbvGrd', 'LowQualFinSF', 'MoSold', 'HouseStyle', 'YrSold', 'MasVnrType', 'Foundation', 'HalfBath', 'FireplaceQu', 'BsmtFinSF2', 'PavedDrive', 'Electrical', 'GarageCond', 'LotShape', 'ExterQual', 'LotConfig', 'Fireplaces', 'EnclosedPorch', 'Exterior1st', 'BsmtExposure', 'GarageFinish', 'BldgType']
#labels = list(train_datax)
train_data = pd.DataFrame()
test_data = pd.DataFrame()

for label in labels:
    train_data[label] = train_datax[label]
    if label != 'SalePrice':
        test_data[label] = test_datax[label]

train_data['SalePrice'] = train_datax['SalePrice']
test_data['Id'] = test_datax['Id']

for label in labels:
    print(label)
    if label == 'Id':
        continue
    if len(train_data.loc[train_data.isnull()[label], label]) > 1000:
        train_data.drop(label, 1, inplace=True)
        if label != 'SalePrice':
            test_data.drop(label, 1, inplace=True)
    else:
        if train_data[label].dtype == np.dtype(object):
            if label != 'SalePrice':
                unq = np.concatenate((train_data[label].unique(),test_data[label].unique()))
            else:
                unq = train_data[label].unique()
            idx1 = 1
            for val in unq:
                train_data.loc[train_data[label] == val, label] = idx1
                if label != 'SalePrice':
                    test_data.loc[test_data[label] == val, label] = idx1
                idx1 = idx1 + 1
        mean_value = np.mean(train_data.loc[train_data.isnull()[label] == False, label])
        train_data.loc[train_data.isnull()[label], label] = mean_value
        if label != 'SalePrice':
            print(label, mean_value)
            test_data.loc[test_data.isnull()[label], label] = mean_value


train_data.to_csv("train_new1.csv")
test_data.to_csv("test_new1.csv")