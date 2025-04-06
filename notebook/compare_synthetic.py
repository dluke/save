# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# fresh notebook for comparing MDL-PWL models on synthetic data

# %% 
import os
join = os.path.join 
import numpy as np
pi = np.pi
norm = np.linalg.norm
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd

from tabulate import tabulate

import _fj

import shapeplot
import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

from pili import support
import mdl

# %% 
# ----------------------------------------------------------------
# config // plotting

def new_plot_synthetic(pwl, synthdata):
    fig, ax = plt.subplots(figsize=(20,5))

    fade_style = {"color":'k', "alpha":0.2}
    ax.plot(pwl.x, pwl.y, **fade_style)

    data_style = {"color":'b', "alpha":0.4, "linestyle":'--', "marker":'o'}
    ax.plot(synthdata.x, synthdata.y, **data_style)
    ax.set_aspect("equal")

def plot_synthetic(ax, pwl):
    fade_style = {"color":'k', "alpha":0.6, "linewidth":"2"}
    ax.plot(pwl.x, pwl.y, **fade_style)


# %% 
# ----------------------------------------------------------------
# 1. adjust sampling frequency

np.random.seed(0)

_l = 0.2
r = 0.03
length = Uniform(_l, _l)
angle = Normal(scale=pi/4)

dx = Constant(_l/4)
error = Normal(scale=r/2)

N = 10
pwl = new_pwl_process(length, angle, N)
synthdata = sample_pwl(pwl, dx, error)


# %% 
# ----------------------------------------------------------------
# 2. adjust noise
