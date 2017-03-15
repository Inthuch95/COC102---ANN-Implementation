'''
Created on Feb 20, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np
from ANN.MLP import MLP
import pandas as pd
from Data.Datasets import datasets
from Data.Preprocessing import data_cleansing
from Data.Preprocessing import remove_outliers
from Data.Preprocessing import standardise

class BackPropagation():
    def __init__(self, dataset, network):
        self.BIAS = 1
        self.dataset = dataset
        self.network = network
        self.s_val = np.array([])
        self.u_val = np.array([])
        self.c = 1
        self.error_val = np.array([])
    
    def forward_pass(self):
        inp = np.array([self.BIAS, 1, 0])
        self.s_val = np.array([])
        self.u_val = np.array([])
        self.u_prime = np.array([])
        # forward pass for one row of features
        for layer in self.network.layers[1:]:
            # empty output from previous layer
            outputs = np.array([])
            for perceptron in layer.perceptrons:
                s = np.sum(perceptron.weights * inp)
                self.s_val = np.append(self.s_val, s)
                outputs = np.append(outputs, self.sigmoid_function(s))
            self.u_val = np.append(self.u_val, outputs)
            inp = np.append([self.BIAS], outputs)
        self.u_prime = np.array([self.sigmoid_function(s, derivative=True) for s in self.s_val])
        
        print("output: ", outputs)
        print("S: ", self.s_val)
        print("U: ", self.u_val)
        print("U': ", self.u_prime)
            
    def backward_pass(self):
        pass
        
    def update(self):
        pass
    
    def sigmoid_function(self, s, derivative=False):
        if derivative:
            return self.sigmoid_function(s) * (1 - self.sigmoid_function(s))
        else:
            return 1/(1 + np.e**-s)
    
if __name__ == "__main__":
    network = MLP(2, 1, 2, 1)
    
    df = pd.read_excel("../Data.xlsx")
    df = df[["AREA", "BFIHOST", "PROPWET", "Index flood"]]
    df = data_cleansing(df)
    df = remove_outliers(df, "PROPWET")
    max_label = np.max(df["Index flood"])
    min_label = np.min(df["Index flood"])
    df = standardise(df)
    # add bias column
    df["BIAS"] = 1
    df = df[["BIAS", "AREA", "BFIHOST", "PROPWET", "Index flood"]]
    features = np.array(df.drop("Index flood", axis=1)) 
    ds = datasets(df, features, "Index flood", max_label, min_label)
    
    network.layers[1].perceptrons[0].weights = [1., 3., 4.]
    network.layers[1].perceptrons[1].weights = [-6., 6., 5.]
    network.layers[2].perceptrons[0].weights = [-3.92, 2., 4.]
    
    clf = BackPropagation(ds, network)
    print("Before training: ", clf.network)
    # perform 1 epoch
    clf.forward_pass()
    clf.backward_pass()
    clf.update()
    #print("After training: ", network)