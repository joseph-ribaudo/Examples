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
import ann

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
    NN = ann.Neural_Network()
    yHat = NN.forward(X)
    print yHat
    print y

# Call the Main Function at Run Time #
if __name__ == "__main__":
    main()
