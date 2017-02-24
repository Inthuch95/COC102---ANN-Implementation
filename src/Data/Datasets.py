'''
Created on Feb 14, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np
from scipy import stats

class datasets():
    def __init__(self, df, features, label):
        self.features = features
        self.label = np.array(df[label])
        self.feature_names = df.drop(label, axis=1).columns.values
        self.label_name = label
        
    def standardise(self, X_train):
        pass

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