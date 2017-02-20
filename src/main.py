'''
Created on Feb 20, 2017

@author: Inthuch Therdchanakul
'''
import pandas as pd
import numpy as np
from Data.Datasets import datasets

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