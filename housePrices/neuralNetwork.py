#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:38:13 2017

@author: r3gz3n
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:12:11 2017

@author: r3gz3n
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv("train_new1.csv")
test_data = pd.read_csv("test_new1.csv")

trainy_data = train_data['SalePrice']
trainx_data = train_data[train_data.columns.difference(['SalePrice', 'Id'])]
testx_data = test_data[test_data.columns.difference(['Id'])]

X = trainx_data.as_matrix()
y = trainy_data.as_matrix()
X_test = testx_data.as_matrix()
"""
scaler = StandardScaler()  
scaler.fit(X)  
X = scaler.transform(X)  
X_test = scaler.transform(X_test)
"""
nn = MLPRegressor(hidden_layer_sizes=(80,), solver='lbfgs', max_iter=2000)
nn.fit(X, y)
pred = nn.predict(X_test)
d = {'Id':test_data['Id'], 'SalePrice':pred}
df = pd.DataFrame(data=d)
df.to_csv('submission_neuralNetwork.csv')
