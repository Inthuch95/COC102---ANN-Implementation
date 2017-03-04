'''
Created on Feb 20, 2017

@author: Inthuch Therdchanaku
'''
import numpy as np
from ANN.MLP import MLP
import pandas as pd
from Data.Datasets import datasets
from Data.Preprocessing import data_cleansing
from Data.Preprocessing import remove_outliers
from Data.Preprocessing import standardise

BIAS = 1

def BackPropagation(dataset, network):
    inputs = np.array([[BIAS, 1, 0]])
    
    # forward pass
    for inp in inputs:
        for layer in network.layers[1:]:
            outputs = np.array([])
            for perceptron in layer.perceptrons:
                tot = np.sum(perceptron.weights * inp)
                outputs = np.append(outputs, sigmoid_function(tot))
            inp = np.array([BIAS])
            inp = np.append(inp, outputs)
        print(outputs)
        
    return network

def sigmoid_function(s):
    return 1/(1 + np.e**-s)

if __name__ == "__main__":
    network = MLP(2, 1, 2, 1)
    
    df = pd.read_excel("../Data.xlsx")
    df = df[["AREA", "BFIHOST", "PROPWET", "Index flood"]]
    df = data_cleansing(df)
    df = remove_outliers(df, "PROPWET")
    max_arr = []
    min_arr = []
    for col in df.columns.values:
        max_arr.append(np.max(df[col]))
        min_arr.append(np.min(df[col]))
    df = standardise(df)
    features = np.array(df.drop("Index flood", axis=1)) 
    ds = datasets(df, features, "Index flood", max_arr, min_arr)
    
    network.layers[1].perceptrons[0].weights = [1, 3, 4]
    network.layers[1].perceptrons[1].weights = [-6, 6, 5]
    network.layers[2].perceptrons[0].weights = [-3.92, 2, 4]
    print("Before training: ", network)
    BackPropagation(ds, network)
    #print("After training: ", network)