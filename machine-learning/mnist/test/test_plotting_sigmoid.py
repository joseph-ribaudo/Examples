#!/usr/bin/python
####################
# Plotting Sigmoid #
############ v1.0 ######
# Author: Abe Hoffman  ###########################
# E-Mail: abraham r hoffman [at] gmail [dot] com #
##################################################################################
# Arizona Artificial Intelligence and Machine Learning Association (AAMA) [2016] #
# LICENSE: Apache 2 License - http://www.apache.org/licenses/LICENSE-2.0.txt     #
##################################################################################

# Import Libraries #
import numpy as np
import matplotlib.pyplot as plt

# Define Global Variables #
x = np.arange(-6., 6., 0.1)

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
    '''
    To see the derivative of Sigmoid:
    Change deriv=False to deriv=True
    '''
    plt.plot(x,sigmoid(x, deriv=False))
    plt.grid(True)
    plt.show()

# Call the Main Function at Run Time #
if __name__ == "__main__":
    main()
