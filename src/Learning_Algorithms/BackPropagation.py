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
import pickle
import matplotlib.pyplot as plt

class BackPropagation():
    def __init__(self, dataset, network):
        self.BIAS = 1
        self.P = 0.1
        self.dataset = dataset
        self.network = network
        self.u = None
        self.prediction = np.array([])
    
    def forward_pass(self, inp):
        # clear previous values
        s_val = np.array([])
        self.u_prime = np.array([])
        # forward pass for one row of features
        for layer in self.network.layers[1:]:
            # empty output from previous layer
            outputs = np.array([])
            for perceptron in layer.perceptrons:
                perceptron.u = inp
                s = np.sum(perceptron.weights * inp)
                s_val = np.append(s_val, s)
                outputs = np.append(outputs, self.sigmoid_function(s))
            inp = np.append([self.BIAS], outputs)
        self.u = outputs[-1]
        self.u_prime = np.array([self.sigmoid_function(s, derivative=True) for s in s_val])
            
    def backward_pass(self, label):
        # propagate deltas backward from output layer to input layer
        self.network.layers[-1].perceptrons[0].delta = (label - self.u) * (self.u_prime[-1])
        # update weights
        self.network.layers[-1].perceptrons[0] = self.update(self.network.layers[-1].perceptrons[0])
        
        output_delta = self.network.layers[-1].perceptrons[0].delta
        # calculate deltas in hidden layer
        for i in range(len(self.network.layers[1].perceptrons),0,-1):
            weight = self.network.layers[-1].perceptrons[0].weights[i]
            self.network.layers[1].perceptrons[i-1].delta = np.sum(weight * output_delta) * (self.u_prime[i-1])
            self.network.layers[1].perceptrons[i-1] = self.update(self.network.layers[1].perceptrons[i-1])
        
    def update(self, perceptron):
        # update every weight linked to the perceptron using deltas
        perceptron.weights = perceptron.weights + (self.P * perceptron.delta * perceptron.u)
        return perceptron
    
    def train(self, epoch=1):
        for i in range(epoch):
            for feature, label in zip(self.dataset.features, self.dataset.label):
                self.forward_pass(feature)
                self.backward_pass(label)
            print("epoch: ", i)
#         print("After %s epoch \n%s" % (epoch, self.network))
#         print("Correct values: ", self.dataset.label[-1])
#         print("Prediction: ", self.prediction)
        
    def predict(self, inp):
        # clear previous values
        s_val = np.array([])
        # forward pass for one row of features
        for layer in self.network.layers[1:]:
            # empty output from previous layer
            output = np.array([])
            for perceptron in layer.perceptrons:
                perceptron.u = inp
                s = np.sum(perceptron.weights * inp)
                s_val = np.append(s_val, s)
                output = np.append(output, self.sigmoid_function(s))
            inp = np.append([self.BIAS], output)
        self.prediction = np.append(self.prediction, output)
    
    def sigmoid_function(self, s, derivative=False):
        if derivative:
            return self.sigmoid_function(s) * (1 - self.sigmoid_function(s))
        else:
            return 1/(1 + np.e**-s)
    
if __name__ == "__main__":
    # change first param to 2 if dummy data is used
    network = MLP(3, 1, 5, 1)
    
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
    
#     network.layers[1].perceptrons[0].weights = np.array([1., 3., 4.])
#     network.layers[1].perceptrons[1].weights = np.array([-6., 6., 5.])
#     network.layers[2].perceptrons[0].weights = np.array([-3.92, 2., 4.])
    
    clf = BackPropagation(ds, network)
    print("Before training: ", clf.network)
    # default is train for 1 epoch
    clf.train(10000)
    x = np.array([idx for idx in range(len(clf.dataset.features))])
    y_observed = clf.dataset.label
    y_modelled = clf.prediction 
    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.plot(x, y_observed, label="Observed")
    ax1.plot(x, y_modelled, color="r", label="Modelled")
    ax1.legend()
    
    ax2 = f2.add_subplot(111)
    ax2.scatter(y_observed, y_modelled)
    plt.show()
#     for _ in range(20000):
#         clf.forward_pass(np.array([clf.BIAS, 1, 0], dtype="float64"))
#         clf.backward_pass(1)
#         clf.update()
#     print(clf.network)
#     print("Correct values: ", 1)
#     print()
#     print("Prediction: ", clf.prediction[-1])
    #print("After training: ", network)