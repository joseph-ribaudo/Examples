
Backpropagation can be separated into 4 distinct sections
1. the forward pass
2. the loss function
3. the backward pass
4. the weight update.

[Forward Pass]   Weights randomized -> output will be something unimportant [.1, .1, .1, .1, .1, .1, .1]
[Loss Function]  [.1, .1, .1, .1, .1, .1, .1] -> [0 0 0 1 0 0 0 0 0 0]
                 A loss function can be defined in many different ways but a common one is MSE
                  (mean squared error), which is ½ times (actual - predicted) squared.




Learning Rate
 learning rate is a parameter that is chosen by the programmer. A high learning rate means that bigger steps are taken in the weight updates and thus, it may take less time for the model to converge on an optimal set of weights. However, a learning rate that is too high could result in jumps that are too large and not precise enough to reach the optimal point.

Epoch
 The process of forward pass, loss function, backward pass, and parameter update is generally called one epoch.
