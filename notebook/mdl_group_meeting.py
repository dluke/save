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
# make figures for group meeting

# %% 
import os
import json
import numpy as np
import scipy.stats
import scipy.optimize
import pickle
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili
from pili import support
import _fj
import mdl
import pwlpartition
import pwlstats

from skimage.restoration import denoise_wavelet
from skimage.restoration import  estimate_sigma

# %% 
mplstyle = {"font.size": 20}

# %% 

# load candidate
path = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
solver = pwlstats.load_solver_at(path)
partition = solver.partition
model = partition.model
data = partition.data

# %%

shortdata = partition.data.cut_index(0,200)
shortmodel = model.cut(0, 200)


# %%

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1, c2, c3 = defcolors[:3] 
model_style = {"linestyle": '-', 'marker': 'D', 'lw':4, 'alpha':0.5, 'color':c2}
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.3, 'color':c3}


def local_simple_plot(ax, model, data):
    ax.plot(data.x, data.y, label='data', **ptlkw)
    ax.plot(model.x, model.y, label='wavelet', **model_style)
    ax.set_aspect('equal')
    # ax.legend(fontsize=20, loc=(1.04, 0))
    ax.legend(fontsize=20, loc=(1.04, 0))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(10,10))
    local_simple_plot(ax, shortmodel, shortdata)

# %%


def plot_partition(ax, model, data):

    color = itertools.cycle(['#FEC216', '#F85D66', '#75E76E'])
    time = model.get_time()
    split = np.split(data.get_n2(), time[:None])
    for sp in split:
        c = next(color)
        x, y = sp.T
        ax.plot(x, y, c=c, linestyle='none', marker='o')
    ax.set_aspect("equal")

    ax.set_xlabel('x')
    ax.set_ylabel('y')

with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(10,10))
    plot_partition(ax, shortmodel, shortdata)


plot_target = join(pwlstats.root, "impress/images")
target = join(plot_target, "candidate_color_partitions.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target, bbox_inches='tight')


# %%


def plot_partition(ax, model, data):

    color = itertools.cycle(['#FEC216', '#F85D66', '#75E76E'])
    time = model.get_time()
    split = np.split(data.get_n2(), time[:None])
    for sp in split:
        c = next(color)
        x, y = sp.T
        ax.plot(x, y, c=c, linestyle='none', marker='o')
    ax.set_aspect("equal")



with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(10,10))
    plot_partition(ax, shortmodel, shortdata)


# %%



with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(10,10))
    plot_partition(ax, shortmodel, shortdata)


# %%
