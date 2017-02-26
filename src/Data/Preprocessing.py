'''
Created on Feb 26, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np

def data_cleansing(df):
    # change all string data to -999 so that they are considered to be outliers and get dropped later on
    for row in range(0, len(df)):
        for col in df.columns.values:
            if isinstance(df[col][row], str):
                df[col][row] = -999 
                
    print("Before preprocessing: ", len(df))
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