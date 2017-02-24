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

def pre_processing(df):
    # change all string data to -999 so that they are considered to be outliers and get dropped later on
    for row in range(0, len(df)):
        for col in df.columns.values:
            if isinstance(df[col][row], str):
                df[col][row] = -999 
                
    print("Before dropping missing data: ", len(df))
    # drop missing data
    df.dropna(inplace=True)
    print("After dropping missing data: ", len(df))
    # eliminate outliers by keeping only the ones that are within 3 standard deviations
    df = df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    print("After dropping outliers: ", len(df))
    return df

def standardise(df):
    for col in df.columns.values:
        df[col] = 0.8 * ((df[col] - np.min(df[col]))/(np.max(df[col]) - np.min(df[col]))) + 0.1
    
    return df