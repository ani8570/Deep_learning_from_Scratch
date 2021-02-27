import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)

def Softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
