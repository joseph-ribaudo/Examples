#!/usr/bin/python

# Import Libraries #
import csv, random

# Main Function #
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
         lines = csv.reader(csvfile)
         dataset = list(lines)
         for x in range(len(dataset)-1):
             for y in range(4):
                 dataset[x][y] = float(dataset[x][y])
                 if random.random() < split:
                    trainingSet.append(dataset[x])
                 else:
                    testSet.append(dataset[x])


if __name__ == "__main__":
        trainingSet=[]
        testSet=[]
        loadDataset('data/iris.csv', 0.66, trainingSet, testSet)
        print 'Train: ' + repr(len(trainingSet))
        print 'Test: ' + repr(len(testSet))


# View Data #
#with open('data/iris.data', 'rb') as csvfile:
#     lines = csv.reader(csvfile)
#     for row in lines:
#         print ', '.join(row)

#lines = csv.reader(csvfile)
