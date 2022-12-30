import numpy as np

def log_loss(predicted, actual):
    tol = 1e-6
    cross_entropy = actual * np.log(predicted + tol)
    return -np.sum(cross_entropy)

def log_loss_grad(predicted, actual):
    return predicted - actual

def softmax(preds):
    tol = 1e-6
    preds = np.exp(preds - np.max(preds))
    return preds / (np.sum(preds) - tol)