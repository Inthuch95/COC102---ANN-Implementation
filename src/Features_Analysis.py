'''
Created on Feb 26, 2017

@author: Inthuch Therdchanakul
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Data.Preprocessing import data_cleansing

df = pd.read_excel("Data.xlsx")
df = data_cleansing(df)
x = [idx for idx in range(len(df))]
y = np.array(df["AREA"])
plt.plot(x, y)
plt.xlabel("Index")
plt.ylabel("AREA")
plt.title("AREA")
plt.show()