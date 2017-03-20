'''
Created on Feb 20, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np
from math import sqrt
import pandas as pd

class BackPropagation():
    def __init__(self, train_set, val_set, test_set, network):
        self.BIAS = 1
        self.P = 0.1
        self.learning_rate = 0.9
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.network = network
        self.u = None
        self.predictions = np.array([])
        self.msre = None
        self.rmse = None
        self.ce = None
        self.rsqr = None
        self.train_rmse = None
        self.train_msre = None
        self.train_ce = None
        self.train_rsqr = None
        self.momentum = False
        self.sa = False
        self.bold_drv = False
    
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
        # Simulated annealing
        # q = learning rate
        # p = step decay
        # r = total number of epoch
        # x = weight?
        
        # update every weight linked to the perceptron using deltas
        if self.momentum:
            # apply momentum
            weight_new = np.array([])
            weight_changed = np.array([])
            weight_new = perceptron.weights + (self.P * perceptron.delta * perceptron.u)
            weight_changed = weight_new - perceptron.weights
            perceptron.weights = weight_new + (self.learning_rate * weight_changed)
        elif self.bold_drv:
            pass
        elif self.sa:
            self.simulated_annealing()
        else:
            perceptron.weights = perceptron.weights + (self.P * perceptron.delta * perceptron.u)
        return perceptron
    
    def simulated_annealing(self):
        print("simulated annealing")
    
    def train(self, epoch=1, momentum=False, sa=False):
        self.msre = np.zeros(epoch)
        self.rmse = np.zeros(epoch)
        self.ce = np.zeros(epoch)
        self.rsqr = np.zeros(epoch)
        
        self.train_msre = np.zeros(epoch)
        self.train_rmse = np.zeros(epoch)
        self.train_ce = np.zeros(epoch)
        self.train_rsqr = np.zeros(epoch)
        self.momentum = momentum
        self.sa = sa
        for i in range(epoch):
            for feature, label in zip(self.train_set.features, self.train_set.label):
                self.forward_pass(feature)
                self.backward_pass(label)
            # calculate validation set error
            self.predict(self.val_set.features)
            self.msre[i] = self.calculate_msre(i, self.val_set.label)
            self.rmse[i] = self.calculate_rmse(i, self.val_set.label)
            self.ce[i] = self.calculate_rsqr(i, self.val_set.label)
            self.rsqr[i] = self.calculate_ce(i, self.val_set.label)
            
            # calculate training set error
            self.predict(self.train_set.features)
            self.train_msre[i] = self.calculate_msre(i, self.train_set.label)
            self.train_rmse[i] = self.calculate_rmse(i, self.train_set.label)
            self.train_ce[i] = self.calculate_rsqr(i, self.train_set.label)
            self.train_rsqr[i] = self.calculate_ce(i, self.train_set.label)
            #if ((i+1) % 2000) == 0:
            print("epoch: ", str(i+1))
        print("Training completed!")
        modelled = pd.DataFrame(self.predictions)
        observed = pd.DataFrame(self.val_set.label)
        
        errors_validation = pd.DataFrame(self.rmse, columns=["RMSE"])
        errors_validation["MSRE"] = self.msre
        errors_validation["CE"] = self.ce
        errors_validation["R2"] = self.rsqr
        modelled.to_csv("modelled.csv", header=False, index=False)
        observed.to_csv("observed.csv", header=False, index=False)
        errors_validation.to_csv("erors_validation.csv", index=False)
        
        errors_train = pd.DataFrame(self.rmse, columns=["RMSE"])
        errors_train["MSRE"] = self.train_msre
        errors_train["CE"] = self.train_ce
        errors_train["R2"] = self.train_rsqr
        errors_validation.to_csv("erors_train.csv", index=False)
    
    # make predictions using trained model     
    def predict(self, features):
        self.predictions = np.array([])
        for feature in features:
            # clear previous values
            s_val = np.array([])
            # forward pass for one row of features
            for layer in self.network.layers[1:]:
                # empty output from previous layer
                output = np.array([])
                for perceptron in layer.perceptrons:
                    perceptron.u = feature
                    s = np.sum(perceptron.weights * feature)
                    s_val = np.append(s_val, s)
                    output = np.append(output, self.sigmoid_function(s))
                feature = np.append([self.BIAS], output)
            self.predictions = np.append(self.predictions, output)
            
    # calculate mean squared relative error from validation set
    def calculate_msre(self, index, data):
        return np.mean(((self.predictions - data)/data)**2)
    
    # calculate root mean squared error from validation set    
    def calculate_rmse(self, index, data):
        return sqrt(np.mean((self.predictions - data)**2))
        
    def calculate_ce(self, index, data):
        topVal = np.sum((self.predictions - data)**2)
        bottomVal = np.sum((data - np.mean(data))**2)
        return 1 - (topVal/bottomVal)
        
    def calculate_rsqr(self, index, data):
        x = self.predictions
        y = data
        x_y = x * y
        x_sqr = x**2
        y_sqr = y **2
        n = len(self.predictions)
        rTop = ((n * np.sum(x_y)) - (np.sum(x) * np.sum(y)))
        rBottom = sqrt((n * np.sum(x_sqr) - (np.sum(x)**2)) * (n * np.sum(y_sqr) - (np.sum(y)**2) ))
        return (rTop/rBottom)**2
    
    # activation function
    def sigmoid_function(self, s, derivative=False):
        if derivative:
            return self.sigmoid_function(s) * (1 - self.sigmoid_function(s))
        else:
            return 1/(1 + np.e**-s)