import numpy as np
from numpy.core.numeric import full

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fully_connected_layer():
    x = np.random.randn(10,2)
    w1 = np.random.randn(2,4)
    b1 = np.random.randn(4)
    w2 = np.random.randn(4,3)
    b2 = np.random.randn(3)

    h = np.matmul(x, w1) + b1
    a = sigmoid(h)
    s = np.matmul(a, w2) + b2
    print(s)
    
def main() :
    fully_connected_layer()

if __name__ == "__main__":
    main()