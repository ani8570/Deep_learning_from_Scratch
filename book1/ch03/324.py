from .func import *
x = np.arange(-5.0, 5.0, 0.1)
y = Sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()