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

df = pd.read_excel("Data.xlsx")
df = df[["AREA", "BFIHOST", "PROPWET", "Index flood"]]
df = data_cleansing(df)
df = remove_outliers(df, "PROPWET")
max_arr = []
min_arr = []
np.min(df["Index flood"])
for col in df.columns.values:
    max_arr.append(np.max(df[col]))
    min_arr.append(np.min(df[col]))
df = standardise(df)
df = df.sample(frac=1).reset_index(drop=True)
# split data set
train, validate, test = np.split(df, [int(.6*len(df)), int(.8*len(df))])
X_train = np.array(train.drop("Index flood", axis=1)) 
train_set = datasets(train, X_train, "Index flood", max_arr, min_arr)
