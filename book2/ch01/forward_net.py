import numpy as np
from util.layer import *
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        I, H, O = input_size, hidden_size, output_size

        w1 = np.random.randn(I,H)
        b1 = np.random.randn(H)
        w2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    


x = np.random.randn(10, 2)
model = TwoLayerNet(2,4,3)
print(x)
s = model.predict(x)
print(s)