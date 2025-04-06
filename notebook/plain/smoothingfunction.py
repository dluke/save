
import numpy as np
import matplotlib.pyplot as plt

def smooth(x):
    return 1 + x**2 * (2*x - 3)

xspace = np.linspace(0,1,1000)

y = [smooth(x) for x in xspace]
plt.plot(xspace, y)
plt.show()