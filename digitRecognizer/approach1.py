import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

train = pd.read_csv("train.csv")

total_X = train.iloc[:5000, 1:]
total_y = train.iloc[:5000, :1]

train_X, test_X, train_y, test_y = train_test_split(total_X, total_y, train_size = 0.8, random_state = 0)

train_X[train_X > 0] = 1
test_X[test_X > 0] = 1

clf = svm.SVC()
clf.fit(train_X, train_y.values.ravel())
print(clf.score(test_X, test_y))

test = pd.read_csv("test.csv")
test[test > 0] = 1
result = clf.predict(test[:5000])

df = pd.DataFrame(result)
df.index.name = "ImageId"
df.index += 1
df.columns = ["Label"]
df.to_csv("mysolution1.csv", header=True)
