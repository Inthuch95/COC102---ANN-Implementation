'''
Created on Feb 14, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np

class datasets():
    def __init__(self):
        self.data = []
        self.target = []
        self.feature_names = []
        self.target_col = ""
    
    def create_data(self, df, label):
        df.fillna(-999, inplace=True)
        self.feature_names = df.drop(label, axis=1).columns.values
        self.target_col = label
        features = np.array(df.drop(label, axis=1))
        for i in range(0, len(features)):
            for j in range(0, len(features[i])):
                if isinstance(features[i][j], str):
                    features[i][j] = -999      
        self.data = features
        self.target = np.array(df[label])