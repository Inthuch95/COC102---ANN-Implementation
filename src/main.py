'''
Created on Feb 14, 2017

@author: Inthuch Therdchanakul
'''
from Datasets import datasets

cwdata = datasets()
cwdata.load_data("Data.xlsx")
print("Label: ", cwdata.target)