import numpy as np

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = np.array([1,2 ,3 ,4, 5, 6, 7, 8, 9])

print(y)

y = y.reshape(1, y.size)
print(y)