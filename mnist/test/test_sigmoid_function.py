#!/usr/bin/python
###################
# Testing Sigmoid #
########## v1.0 ########
# Author: Abe Hoffman  ###########################
# E-Mail: abraham r hoffman [at] gmail [dot] com #
##################################################################################
# Arizona Artificial Intelligence and Machine Learning Association (AAMA) [2016] #
# LICENSE: Apache 2 License - http://www.apache.org/licenses/LICENSE-2.0.txt     #
##################################################################################

# Import Libraries #
import numpy as np

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
    print
    print "Sigmoid(1)   :  %s" % sigmoid(1)
    print "Sigmoid(2)   :  %s" % sigmoid(2)
    print "Sigmoid(3)   :  %s" % sigmoid(3)
    print "Sigmoid(28)  :  %s" % sigmoid(28)
    print "Sigmoid(29)  :  %s" % sigmoid(29)
    print
    print "Sigmoid'(1)  :  %s" % sigmoid(1.0, deriv=True)
    print "Sigmoid'(2)  :  %s" % sigmoid(2.0, deriv=True)
    print "Sigmoid'(3)  :  %s" % sigmoid(3.0, deriv=True)
    print "Sigmoid'(28) :  %s" % sigmoid(28.0, deriv=True)
    print "Sigmoid'(29) :  %s" % sigmoid(29.0, deriv=True)
    print

# Call the Main Function at Run Time #
if __name__ == "__main__":
    main()
