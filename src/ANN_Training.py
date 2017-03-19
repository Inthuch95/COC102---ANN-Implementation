'''
Created on Feb 20, 2017

@author: Inthuch Therdchanakul
'''
import pandas as pd
import numpy as np
from Data.Datasets import datasets
from Data.Preprocessing import data_cleansing
from Data.Preprocessing import remove_outliers
from Data.Preprocessing import standardise
from Learning_Algorithms.BackPropagation import BackPropagation
from ANN.MLP import MLP
import matplotlib.pyplot as plt
import pickle

df = pd.read_excel("Data.xlsx")
df = df[["AREA", "BFIHOST", "PROPWET", "Index flood"]]
df = data_cleansing(df)
df = remove_outliers(df, "PROPWET")
max_label = np.max(df["Index flood"])
min_label = np.min(df["Index flood"])
df = standardise(df)
df["BIAS"] = 1
df = df[["BIAS", "AREA", "BFIHOST", "PROPWET", "Index flood"]]
df = df.sample(frac=1).reset_index(drop=True)
# split data set
df_train, df_val, df_test = np.split(df, [int(.6*len(df)), int(.8*len(df))])
features_train = np.array(df_train.drop("Index flood", axis=1)) 
features_val = np.array(df_val.drop("Index flood", axis=1))
features_test = np.array(df_test.drop("Index flood", axis=1))

train_set = datasets(df_train, features_train, "Index flood", max_label, min_label)
val_set = datasets(df_val, features_val, "Index flood", max_label, min_label)
test_set = datasets(df_test, features_test, "Index flood", max_label, min_label)

network = MLP(3, 1, 5, 1)
clf = BackPropagation(train_set, val_set, test_set, network)
epoch = 20000
clf.train(epoch, momentum=True)
clf.predict(test_set.features)

filename = input("Enter file name: ")
filename = "".join([filename, ".pickle"]) 
with open(filename, "wb") as f:
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
    
x = np.array([idx for idx in range(len(test_set.features))])
y_observed = test_set.label
y_modelled = clf.predictions
f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(x, y_observed, label="Observed")
ax1.plot(x, y_modelled, color="r", label="Modelled")
ax1.legend()

ax2 = f2.add_subplot(111)
ax2.scatter(y_observed, y_modelled)

ax3 = f3.add_subplot(111)
x = np.array([idx for idx in range(epoch)])
y = clf.msre
ax3.plot(x, y)

ax4 = f4.add_subplot(111)
y = clf.rmse
ax4.plot(x, y)
plt.show()