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
import os

df = pd.read_excel("Data.xlsx")
df = df[["AREA", "BFIHOST", "PROPWET", "Index flood"]]
df = data_cleansing(df)
df = remove_outliers(df, "AREA", threshold=2.5)
df = remove_outliers(df, "BFIHOST")
df = remove_outliers(df, "PROPWET")
df = remove_outliers(df, "Index flood", threshold=2.5)
max_label = np.max(df["Index flood"])
min_label = np.min(df["Index flood"])
df = standardise(df)
df["BIAS"] = 1
df = df[["BIAS", "AREA", "BFIHOST", "PROPWET", "Index flood"]]
# split data set
df_train, df_val, df_test = np.split(df, [int(.6*len(df)), int(.8*len(df))])

features_train = np.array(df_train.drop("Index flood", axis=1)) 
features_val = np.array(df_val.drop("Index flood", axis=1))
features_test = np.array(df_test.drop("Index flood", axis=1))

train_set = datasets(df_train, features_train, "Index flood", max_label, min_label)
val_set = datasets(df_val, features_val, "Index flood", max_label, min_label)
test_set = datasets(df_test, features_test, "Index flood", max_label, min_label)
y_observed = test_set.de_standardise(test_set.label)
# hidden unit value [2, 10]
start = timer()
network = MLP(3, 1, 9, 1)
clf = BackPropagation(train_set, val_set, test_set, network)
training_type = "momentum"
n_hidden_units = str(clf.network.n_perceptrons_to_hl)
print("Trainning Started with %s hidden unit" % str(n_hidden_units))
print("Training mode: ", training_type)
clf.train(momentum=True)
end = timer()  
print("Tranining completed in %s with %s hidden unit" % (str(end-start), str(n_hidden_units)))
y_val = clf.rmse
y_train = clf.train_rmse
# make predictions on testing set
clf.predict(test_set.features)

f1 = plt.figure()
f2 = plt.figure()
x = np.array([idx for idx in range(clf.epoch)])

ax1 = f1.add_subplot(111)
ax1.set_title("RMSE")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Errors")
ax1.plot(x, y_val, label="validation set")
ax1.plot(x, y_train, color="r", label="train set")
ax1.legend()

ax2 = f2.add_subplot(111)
x = np.array([idx for idx in range(len(test_set.label))])
y_modelled = test_set.de_standardise(clf.predictions)
ax2.set_title("Actual vs Predicted")
ax2.set_xlabel("Index")
ax2.set_ylabel("Actual/Predicted")
ax2.plot(x, y_modelled, color="r", label="Predicted")
ax2.scatter(x, y_observed, label="Actual")
ax2.set_ylim(-100, 900)
ax2.legend()

# save the model for analysis later on
path = "Simulations/" + training_type + "_" + n_hidden_units + "_e" + str(clf.epoch) + "/"
if not os.path.exists(path):
    os.makedirs(path)
predictions_filename = path + training_type + "_" + n_hidden_units + "_e" + str(clf.epoch) + "_PREDICTIONS.pickle"
rmse_filename = path + training_type + "_" + n_hidden_units + "_e" + str(clf.epoch) + "_RMSE.pickle"
clf_filename = path + training_type + "_" + n_hidden_units + "_e" + str(clf.epoch) + "_MODEL.pickle"
with open(predictions_filename, "wb") as f:
    pickle.dump(ax2, f, pickle.HIGHEST_PROTOCOL)
with open(rmse_filename, "wb") as f:
    pickle.dump(ax1, f, pickle.HIGHEST_PROTOCOL)  
with open(clf_filename, "wb") as f:
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL) 
plt.show()