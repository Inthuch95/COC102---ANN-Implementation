'''
Created on Mar 18, 2017

@author: Inthuch Therdchanakul    
'''
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Data.Datasets import datasets
from Data.Preprocessing import data_cleansing
from Data.Preprocessing import remove_outliers

training_type = "annealing"
n_hidden_units = "10"
epoch = "11000"

df = pd.read_excel("Data.xlsx")
df = df[["AREA", "BFIHOST", "PROPWET", "Index flood"]]
df = data_cleansing(df)
df = remove_outliers(df, "AREA", threshold=2.5)
df = remove_outliers(df, "BFIHOST")
df = remove_outliers(df, "PROPWET")
df = remove_outliers(df, "Index flood", threshold=2.5)
max_label = np.max(df["Index flood"])
min_label = np.min(df["Index flood"])
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

path = "Simulations/" + training_type + "_" + n_hidden_units + "_e" + epoch + "/"
predictions_filename = path + training_type + "_" + n_hidden_units + "_e" + epoch + "_PREDICTIONS.pickle"
rmse_filename = path + training_type + "_" + n_hidden_units + "_e" + epoch + "_RMSE.pickle"
clf_filename = path + training_type + "_" + n_hidden_units + "_e" + epoch + "_MODEL.pickle"
with open(predictions_filename, "rb") as f:
    predictions = pickle.load(f)
with open(rmse_filename, "rb") as f:
    rmse = pickle.load(f)  
with open(clf_filename, "rb") as f:
    clf = pickle.load(f)   

predictions.set_ylim(-100,900)
test_set = test_set.label
rmse_test = clf.calculate_rmse(test_set)
correl = np.corrcoef(test_set, clf.predictions)[0][1]

print("Instance: ", len(test_set))
print("Correlation coefficient: ", correl)
print("RMSE: ", rmse_test)
best_rmse = min(clf.rmse)
idx = np.argmin(clf.rmse)
print("epoch: ", idx+1)
rmse.scatter(idx, best_rmse, color="m", marker="*", s=50)

final_df = pd.DataFrame({"Observed":test_set, "Modelled":clf.predictions})
final_df.to_csv("Final model.csv", index=False)
plt.show()