'''
Created on Feb 20, 2017

@author: Inthuch Therdchanakul
'''
from Data.Datasets import datasets
import pandas as pd

df = pd.read_excel("Data.xlsx")
df = df[["AREA", "LDP","PROPWET", "RMED-1D", "SAAR", "Index flood"]]
ds = datasets()
ds.create_data(df, "Index flood")