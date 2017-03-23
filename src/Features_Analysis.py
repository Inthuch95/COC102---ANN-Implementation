'''
Created on Feb 26, 2017

@author: Inthuch Therdchanakul
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Data.Preprocessing import data_cleansing
from Data.Preprocessing import remove_outliers

df = pd.read_excel("Data.xlsx")
df = df[["AREA", "BFIHOST", "PROPWET", "Index flood"]]
df = data_cleansing(df)

area = np.array(df["AREA"])
bfihost = np.array(df["BFIHOST"])
propwet = np.array(df["PROPWET"])
idx_flood = np.array(df["Index flood"])

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()

x = np.array([idx for idx in range(len(area))])
    
ax1 = f1.add_subplot(111)
y = area
ax1.set_title("AREA")
ax1.set_xlabel("INDEX")
ax1.set_ylabel("AREA")
ax1.scatter(x, y)

ax2 = f2.add_subplot(111)
y = bfihost
ax2.set_title("BFIHOST")
ax2.set_xlabel("INDEX")
ax2.set_ylabel("BFIHOST")
ax2.scatter(x, y)

ax3 = f3.add_subplot(111)
y = propwet
ax3.set_title("PROPWET")
ax3.set_xlabel("INDEX")
ax3.set_ylabel("PROPWET")
ax3.scatter(x, y)

ax4 = f4.add_subplot(111)
y = idx_flood
ax4.set_title("Index flood")
ax4.set_xlabel("INDEX")
ax4.set_ylabel("Index flood")
ax4.scatter(x, y)
plt.show()

df = remove_outliers(df, "AREA", threshold=2.5)
df = remove_outliers(df, "BFIHOST")
df = remove_outliers(df, "PROPWET")
df = remove_outliers(df, "Index flood", threshold=2.5)
area = np.array(df["AREA"])
bfihost = np.array(df["BFIHOST"])
propwet = np.array(df["PROPWET"])
idx_flood = np.array(df["Index flood"])

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()

x = np.array([idx for idx in range(len(area))])
    
ax1 = f1.add_subplot(111)
y = area
ax1.set_title("AREA")
ax1.set_xlabel("INDEX")
ax1.set_ylabel("AREA")
ax1.set_ylim(0,4000)
ax1.scatter(x, y)

ax2 = f2.add_subplot(111)
y = bfihost
ax2.set_title("BFIHOST")
ax2.set_xlabel("INDEX")
ax2.set_ylabel("BFIHOST")
ax2.scatter(x, y)

ax3 = f3.add_subplot(111)
y = propwet
ax3.set_title("PROPWET")
ax3.set_xlabel("INDEX")
ax3.set_ylabel("PROPWET")
ax2.set_ylim(0,2)
ax3.scatter(x, y)

ax4 = f4.add_subplot(111)
y = idx_flood
ax4.set_title("Index flood")
ax4.set_xlabel("INDEX")
ax4.set_ylabel("Index flood")
ax4.set_ylim(0,1000)
ax4.scatter(x, y)
plt.show()