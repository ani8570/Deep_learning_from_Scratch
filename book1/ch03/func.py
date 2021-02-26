import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)