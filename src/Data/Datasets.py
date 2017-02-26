'''
Created on Feb 14, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np

class datasets():
    def __init__(self, df, features, label, max_arr, min_arr):
        self.features = features
        self.label = np.array(df[label])
        self.feature_names = df.drop(label, axis=1).columns.values
        self.label_name = label
        self.max = np.asarray(max_arr)
        self.min = np.asarray(min_arr)
    
    def de_standardise(self):
        num_col = self.features.shape[1]
        for col in range(0, num_col):
            self.features[col, :] = (((self.features[col, :] - 0.1)/(0.8)) * 
                                     (self.max[:-1] - self.min[:-1])) + self.min[:-1]
        for i in range(0, len(self.label)):
            self.label[i] = (((self.label[i] - 0.1)/(0.8)) * (self.max[-1] - self.min[-1])) + self.min[-1]