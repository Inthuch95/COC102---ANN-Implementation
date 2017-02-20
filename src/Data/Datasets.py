'''
Created on Feb 14, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np

class datasets():
    def __init__(self, df, features, label):
        self.features = features
        self.label = np.array(df[label])
        self.feature_names = df.drop(label, axis=1).columns.values
        self.label_name = label
        
    def standardise(self, X_train):
        pass