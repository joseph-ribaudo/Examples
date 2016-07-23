#!/usr/bin/python
########################
# A XOR Neural Network #
############### v1.0 ###
# Author: Abe Hoffman  ###########################
# E-Mail: abraham r hoffman [at] gmail [dot] com #
##################################################################################
# Arizona Artificial Intelligence and Machine Learning Association (AAMA) [2016] #
# LICENSE: Apache 2 License - http://www.apache.org/licenses/LICENSE-2.0.txt     #
##################################################################################

### Import Libraries ###
import numpy as np

### Define Global Variables ###
# XOR Truth Table:
# input | output
# --------------
# 0, 0  | 0
# 0, 1  | 1
# 1, 0  | 1
# 1, 1  | 0

# Define 'x' as arrays representing inputs for our XOR operation #
x = np.array([ [0,0],
               [0,1],
               [1,0],
               [1,1] ])

# Next, define y as the expected result of our XOR operation #
y = np.array([0,
              1,
              1,
              0]).T

### Define Functions ###

# Activation Function: Sigmoid #
def sigmoid(x, deriv=False):
    '''
    We'll define our Sigmoid Function here.
    Calculate Sigmoid's derivative if deriv=True;
    this is for back propagation of our network.
    '''
    # Calculate the Derivative if deriv=True #
    if (deriv==True):
        return x * (1-x)
    # Otherwise just return f(x)=(sigmoid(x)) #
    return 1.0 / (1.0 + np.exp(-x))

# Main #
def main():
    print sigmoid(1.0, deriv=False)
    print sigmoid(1.0, deriv=True)

# Call the Main Function at Run Time #
if __name__ == "__main__":
    main()
