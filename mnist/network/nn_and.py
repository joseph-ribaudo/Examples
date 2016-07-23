#!/usr/bin/python
##################################
# Neural Network (AND) Operation #
######################### v1.0 ###
# Author: Abe Hoffman  ###########################
# E-Mail: abraham r hoffman [at] gmail [dot] com #
##################################################################################
# Arizona Artificial Intelligence and Machine Learning Association (AAMA) [2016] #
# LICENSE: Apache 2 License - http://www.apache.org/licenses/LICENSE-2.0.txt     #
##################################################################################

### Import Libraries ###
import numpy as np

### Define Classes / Functions ###

# Aritificial Neural Network Class #

class Neural_Network(object):
    def __init__(self):
        ## Define Hyperparameters ##
        # Layer 0
        self.inputLayerSize = 2
        # Layer 1
        self.hiddenLayerSize = 30
        # Layer 2
        self.outputLayerSize = 1

        ## Weights ##
        # Weights for synapses between L0 and L1 Neurons #
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        # Weights for synapses between L1 and L2 Neurons #
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        # ** Note that these weights are initially randomized ** #

    def forward(self, X):
        # Propagate inputs through our network #
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector or matrix #
        return 1 / (1 + np.exp(-z))

# Main #
def main():
    # Define Data #
    X = np.array(([0,0],
                  [0,1],
                  [1,0],
                  [1,1]), dtype=float)
    y = np.array(([0],
                  [0],
                  [0],
                  [1]), dtype=float)

    # Normalize Data #
    #X = X/np.amax(X, axis=0)
    #y = y/100

    # Test Neural Network #
    NN = Neural_Network()
    yHat = NN.forward(X)
    print yHat
    print y

# Call the Main Function at Run Time #
if __name__ == "__main__":
    main()
