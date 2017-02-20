'''
Created on Feb 20, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np

def sigmoid_function(s):
    return 1/(1 + np.e**-s) 