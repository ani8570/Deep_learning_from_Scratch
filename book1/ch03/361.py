import tensorflow as tf
import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img = x_train[0]
label = y_train[0]

print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)