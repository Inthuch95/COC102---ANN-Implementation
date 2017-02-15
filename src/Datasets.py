'''
Created on Feb 14, 2017

@author: Inthuch Therdchanakul
'''
import pandas as pd
import numpy as np

class datasets():
    def __init__(self):
        self.data = []
        self.target = []
        self.feature_names = []
        self.target_col = ""
    
    def load_data(self, data_file):
        df = pd.read_excel(data_file)
        self.feature_names = df.columns.values[:-1]
        self.target_col = "Index flood"
        self.target = df["Index flood"].values
        df.drop('Index flood', axis=1, inplace=True)
        self.data = df.values