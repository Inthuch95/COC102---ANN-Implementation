'''
Created on Feb 14, 2017

@author: Inthuch Therdchanakul
'''
from Data.Datasets import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_excel("Data.xlsx")
df = df[["AREA", "LDP","PROPWET", "RMED-1D", "SAAR", "Index flood"]]
df.fillna(-999, inplace=True)
features = np.array(df.drop("Index flood", axis=1))
for i in range(0, len(features)):
    for j in range(0, len(features[i])):
        if isinstance(features[i][j], str):
            features[i][j] = -999  
ds = datasets(df, features, "Index flood")
print(ds.feature_names)

X = ds.features
y = ds.label
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(y_train, type(y_train))

clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
predictions = clf.predict(X_test)
print(accuracy)