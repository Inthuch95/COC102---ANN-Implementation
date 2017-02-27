'''
Created on Feb 26, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np

def data_cleansing(df):
    # change all string data to -999 so that they are considered to be outliers and get dropped later on 
    df = df[~df.applymap(lambda x: isinstance(x, str))]  
    print("Before preprocessing: ", len(df))
    # drop missing data
    df.dropna(inplace=True)
    print("After dropping missing data: ", len(df))
    # drop negative values
    df = df[df.apply(lambda x: x >= 0).all(axis=1)]
    print("After dropping negative values: ", len(df))
    
    return df

def remove_outliers(df, feature):
    # eliminate outliers by keeping only the ones that are within 3 standard deviations
    df = df[((df[feature] - np.mean(df[feature])) / np.std(df[feature])).abs() < 3]
    
    return df

def standardise(df):
    for col in df.columns.values:
        df[col] = 0.8 * ((df[col] - np.min(df[col]))/(np.max(df[col]) - np.min(df[col]))) + 0.1
    
    return df