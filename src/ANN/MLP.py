'''
Created on Feb 20, 2017

@author: Inthuch Therdchanakul
'''
import random
import numpy as np

BIAS = -1

class Perceptron():
    def __init__(self, n_inputs ):
        self.n_inputs = n_inputs
        self.set_weights( np.array([random.uniform(-2./n_inputs,2./n_inputs) for x in range(0,n_inputs+1)])) # +1 for bias weight

    def set_weights(self, weights ):
        self.weights = weights

    def __str__(self):
        return 'Weights: %s, Bias: %s' % ( str(self.weights[1:]),str(self.weights[0]) )

class PerceptronLayer():
    def __init__(self, n_perceptrons, n_inputs):
            self.n_perceptrons = n_perceptrons
            self.perceptrons = np.array([Perceptron( n_inputs ) for _ in range(0,self.n_perceptrons)])

    def __str__(self):
        return 'Layer:\n\t'+'\n\t'.join([str(perceptron) for perceptron in self.perceptrons])+''

class MLP():
    def __init__(self, n_inputs, n_outputs, n_perceptrons_to_hl, n_hidden_layers):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.n_perceptrons_to_hl = n_perceptrons_to_hl

        # Do not touch
        self._create_network()
        self._n_weights = None
        # end

    def _create_network(self):
        if self.n_hidden_layers>0:
            # create the first layer
            # ignore the weights here!
            self.layers = [PerceptronLayer( self.n_perceptrons_to_hl,self.n_inputs )]

            # create hidden layers
            self.layers += [PerceptronLayer( self.n_perceptrons_to_hl,self.n_perceptrons_to_hl ) for _ in range(0,self.n_hidden_layers)]

            # hidden-to-output layer
            self.layers += [PerceptronLayer( self.n_outputs,self.n_perceptrons_to_hl )]
        else:
            # If we don't require hidden layers
            self.layers = [PerceptronLayer( self.n_outputs,self.n_inputs )]
        self.layers = np.asarray(self.layers)

    def __str__(self):
        return '\n'.join([str(i+1)+' '+str(layer) for i,layer in enumerate(self.layers)])

if __name__ == "__main__":
    network = MLP(2, 1, 2, 1)
    print(network)
