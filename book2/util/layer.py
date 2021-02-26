import numpy as np
from .func import *
class Sigmoid:
    def __init__(self):
        self.params = []
    
    def forward(self, x):
        return 1/(1+np.exp(-x))

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
    
    def forward(self, x):
        w, b = self.params
        out = np.matmul(x, w) + b
        return out

class Softmax:
    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.out = None
    
    # def forward(self, x):
    #     self.out = 

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)


