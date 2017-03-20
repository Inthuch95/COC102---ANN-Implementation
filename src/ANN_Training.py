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
from timeit import default_timer as timer


start = timer()


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
epoch = 1000
clf.train(epoch, momentum=True)

end = timer()
print(end - start)

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()
x = np.array([idx for idx in range(epoch)])

ax1 = f1.add_subplot(111)
y_val = clf.rmse
y_train = clf.train_rmse
ax1.set_title("RMSE")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Errors")
ax1.plot(x, y_val, label="validation set")
ax1.plot(x, y_train, color="r", label="train set")
ax1.legend()

ax2 = f2.add_subplot(111)
y_val = clf.msre
y_train = clf.train_msre
ax2.set_title("MSRE")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Errors")
ax2.plot(x, y_val, label="validation set")
ax2.plot(x, y_train, color="r", label="train set")
ax2.legend()

ax3 = f3.add_subplot(111)
y_val = clf.ce
y_train = clf.train_ce
ax3.set_title("CE")
ax3.set_xlabel("Epochs")
ax3.set_ylabel("Errors")
ax3.plot(x, y_val, label="validation set")
ax3.plot(x, y_train, color="r", label="train set")
ax3.legend()

ax4 = f4.add_subplot(111)
y_val = clf.rsqr
y_train = clf.train_rsqr
ax4.set_title("Rsqr")
ax4.set_xlabel("Epochs")
ax4.set_ylabel("Errors")
ax4.plot(x, y_val, label="validation set")
ax4.plot(x, y_train, color="r", label="train set")
ax4.legend()
plt.show()
# x = np.array([idx for idx in range(len(test_set.features))])
# y_observed = test_set.de_standardise(test_set.label)
# y_modelled = test_set.de_standardise(clf.predictions)
# f1 = plt.figure()
# f2 = plt.figure()
# f3 = plt.figure()
# f4 = plt.figure()
# f5 = plt.figure()
# f6 = plt.figure()
# ax1 = f1.add_subplot(111)
# ax1.set_xlabel("Index")
# ax1.set_ylabel("Actual/Predicted")
# ax1.plot(x, y_observed, label="Actual Values")
# ax1.plot(x, y_modelled, color="r", label="Predicted Values")
# ax1.legend()
# 
# ax2 = f2.add_subplot(111)
# ax2.scatter(y_observed, y_modelled)
# 
# ax3 = f3.add_subplot(111)
# x = np.array([idx for idx in range(epoch)])
# y = clf.msre
# ax3.plot(x, y)
# 
# ax4 = f4.add_subplot(111)
# y = clf.rmse
# ax4.plot(x, y)
# 
# ax5 = f5.add_subplot(111)
# y = clf.rsqr
# ax5.plot(x, y)
# 
# ax6 = f6.add_subplot(111)
# y = clf.ce
# ax6.plot(x, y)
# plt.show()
# clf.predict(test_set.features)
# clf.calculate_msre(0, test_set.label)
# clf.calculate_rmse(0, test_set.label)
# clf.calculate_rsqr(0, test_set.label)
# clf.calculate_ce(0, test_set.label)
# print("RMSE: ", clf.rmse[0])
# print("MSRE: ", clf.msre[0])
# print("CE: ", clf.ce[0])
# print("RSQR: ", clf.rsqr[0])
# modelled = pd.DataFrame(clf.predictions)
# modelled.to_csv("modelled.csv", header=False, index=False)
# observed = pd.DataFrame(test_set.label)
# observed.to_csv("observed.csv", header=False, index=False)


filename = input("Enter file name: ")
filename = "".join([filename, ".pickle"]) 
with open(filename, "wb") as f:
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
    
    
    
    
    
    
