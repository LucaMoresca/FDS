import numpy as np


def sigmoid(x):
    """
    Function to compute the sigmoid of a given input x.

    Args:
        x: it's the input data matrix.

    Returns:
        g: The sigmoid of the input x
    """
    ##############################
    ###     YOUR CODE HERE     ###
    ##############################    
    g = 1 / (1 + np.exp(-x))
    return g

def softmax(y):
    """
    Function to compute associated probability for each sample and each class.

    Args:
        y: the predicted 

    Returns:
        softmax_scores: it's the matrix containing probability for each sample and each class. The shape is (N, K)
    """
    ##############################
    ###     YOUR CODE HERE     ###
    ##############################
    
    
    
    y_st = y - np.max(y, axis=1, keepdims=True)
    
    exps = np.exp(y_st)
    

    
    
    if np.any(np.isnan(exps)) or np.any(np.isinf(exps)):
        print("Warning: Infinite or NaN values detected in softmax computation")
    
    
    
    
    softmax_scores = exps / np.sum(exps, axis=1, keepdims=True)

    return softmax_scores

