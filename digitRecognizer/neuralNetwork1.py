from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

print('Loading and Visualizing Data ...')

train = pd.read_csv("train_testdata.csv")

print('Extracting Features and Label ...')
y = train["label"]
X = train[train.columns.difference(['label'])]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(35, 1), random_state=1)
clf.fit(X, y)

print("Predicting values ...")

pred = clf.predict(X)

pred = pd.DataFrame(pred, index=range(1, pred.shape[0]+1))

pred.to_csv("my_solution1.csv")

print((pred == y).mean() * 100)
