'''
Created on Feb 14, 2017

@author: Inthuch Therdchanakul
'''
import pandas as pd

class datasets():
    def __init__(self):
        self.data = []
        self.target = []
        self.feature_names = []
    
    def load_data(self, data_file):
        df = pd.read_excel(data_file)
        self.feature_names = [df.keys()[key] for key in range(0, len(df.keys())-1)]
        self.target = df[df.keys()[-1]].values