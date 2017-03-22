'''
Created on Feb 20, 2017

@author: Inthuch Therdchanakul
'''
import numpy as np
from math import sqrt

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
        self.rmse = np.array([])
        self.ce = None
        self.rsqr = None
        self.train_rmse = np.array([])
        self.train_msre = None
        self.train_ce = None
        self.train_rsqr = None
        self.momentum = False
        self.sa = False
        self.bold_drv = False
        self.current_epoch = 0
        self.epoch = 0
        self.max_lr = None
        self.min_lr = None
        self.min_error = 1
        self.best_network = network
    
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
        if self.momentum:
            # apply momentum
            weight_old = np.array([])
            weight_old = perceptron.weights
            perceptron.weights = perceptron.weights + (self.P * perceptron.delta * perceptron.u) + (self.learning_rate * perceptron.delta_weights) 
            perceptron.delta_weights = perceptron.weights - weight_old
        elif self.bold_drv:
            print("BOLD DRIVER DOES NOT WORK!")
        elif self.sa:
            # Simulated annealing
            # p = minimum learning rate
            # q = maximum learning rate
            # r = epochs
            # x = current epoch
            r = 15000
            exp = 10 - ((20*self.current_epoch)/r)
            self.learning_rate = self.min_lr + (self.max_lr - self.min_lr) * (1 - (1/(1 + np.e**exp)))
            weight_old = np.array([])
            weight_old = perceptron.weights
            perceptron.weights = perceptron.weights + (self.P * perceptron.delta * perceptron.u) + (self.learning_rate * perceptron.delta_weights) 
            perceptron.delta_weights = perceptron.weights - weight_old
        else:
            perceptron.weights = perceptron.weights + (self.P * perceptron.delta * perceptron.u)
        return perceptron
    
    def train(self, epoch=1000, momentum=False, sa=False, max_lr=1., min_lr=0.01):
        self.momentum = momentum
        self.sa = sa
        self.epoch = self.epoch + epoch
        self.max_lr = max_lr
        self.min_lr = min_lr
        for i in range(epoch):
            self.current_epoch = (self.epoch - epoch) + i
            for feature, label in zip(self.train_set.features, self.train_set.label):
                self.forward_pass(feature)
                self.backward_pass(label)
                
            # calculate training set error
            self.predict(self.train_set.features)
            self.train_rmse = np.append(self.train_rmse, [self.calculate_rmse(self.train_set.label)])
            # calculate validation set error
            self.predict(self.val_set.features)
            rmse = self.calculate_rmse(self.val_set.label)
            self.rmse = np.append(self.rmse, [rmse])
            if np.min(self.rmse) == rmse:
                self.best_network = self.network
            #if ((i+1) % 2000) == 0:
            print("epoch: ", str(self.current_epoch+1))
        min_error = np.min(self.rmse)
        print([self.min_error, min_error])
        if min_error < self.min_error:
            self.min_error = min_error
            self.train(epoch=epoch, momentum=self.momentum, sa=self.sa)  
        self.min_error = min_error
        self.network = self.best_network
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
    def calculate_rmse(self, data):
        #return sqrt(np.mean((self.predictions - data)**2))
        return np.sqrt(np.mean((self.predictions-data)**2))
        
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