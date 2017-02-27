'''
Created on Feb 26, 2017

@author: Inthuch Therdchanakul
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Data.Preprocessing import data_cleansing
from Data.Preprocessing import remove_outliers

def visualise(feature):
    df = pd.read_excel("Data.xlsx")
    df = data_cleansing(df)
    ylim = np.max(df[feature])
    df = remove_outliers(df, feature)
    x = [idx for idx in range(len(df))]
    y = np.array(df[feature])  
    plt.ylim(0, ylim)
    plt.scatter(x, y)
    plt.xlabel("Index")
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()
    
if __name__ == "__main__":
    visualise("PROPWET")