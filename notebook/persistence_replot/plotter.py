
import sys, os

import thesis.publication as thesis

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import axes3d

X = np.loadtxt('X.dat')
Y = np.loadtxt('Y.dat')
Z = np.loadtxt('Z.dat')

print(X.shape, Y.shape, Z.shape)

style = thesis.texstyle.copy()
style["text.usetex"] = False
with mpl.rc_context(style):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    ax.plot_wireframe(X,Y,Z, lw=1)
    # ax.plot_surface(X,Y,Z)

    ax.set_xlabel('persistence')
    ax.set_ylabel('activity')
    ax.set_zlabel(r'P')

plt.show()