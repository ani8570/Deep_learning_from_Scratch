from func import *
from mnist import load_mnist
import numpy as np
import pickle
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
    return x_test, y_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = Sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = Sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = Softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt +=1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))


batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
