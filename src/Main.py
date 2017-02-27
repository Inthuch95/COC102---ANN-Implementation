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
df = df[["AREA", "LDP","PROPWET", "RMED-1D", "SAAR", "Index flood"]]
df = data_cleansing(df)
df = remove_outliers(df, "PROPWET")
max_arr = []
min_arr = []
for col in df.columns.values:
    max_arr.append(np.max(df[col]))
    min_arr.append(np.min(df[col]))
df = standardise(df)
features = np.array(df.drop("Index flood", axis=1)) 
ds = datasets(df, features, "Index flood", max_arr, min_arr)