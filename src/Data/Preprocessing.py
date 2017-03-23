'''
Created on Feb 26, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np

def data_cleansing(df):
    # change all string data to -999 so that they will be removed along with other negative values
    df = df[~df.applymap(lambda x: isinstance(x, str))]  
    # drop missing data
    df.dropna(inplace=True)
    # drop negative values
    df = df[df.apply(lambda x: x >= 0).all(axis=1)]
    return df

def remove_outliers(df, feature, threshold=3):
    # z-score
    # eliminate outliers by keeping only the ones that are within t standard deviations
    # where t is a threshold variable
    df = df[((df[feature] - np.mean(df[feature])) / np.std(df[feature])).abs() < threshold]
    
    return df

def standardise(df):
    for col in df.columns.values:
        df[col] = 0.8 * ((df[col] - np.min(df[col]))/(np.max(df[col]) - np.min(df[col]))) + 0.1
    
    return df