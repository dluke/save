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
# analyse partially solved models to check for convergence
# this module doesn't do any hard computation itself

# %% 
import os
import json
import numpy as np
from glob import glob
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili
import _fj
import mdl
import pwlpartition

import fjanalysis
import pwlstats

# %% 
# setup test data
# this test data generated with pwlpartition.percolate_cleanup

target = join(pwlstats.root, "run/partition/trial/_candidate_pwl")
def load_at(target):
    assert(os.path.exists(target))
    form = join(target, 'iteration_*.pkl')
    itersolver = [pwlpartition.Solver.load_state(path) for path in sorted(glob(form))]
    print("found {} objects".format(len(itersolver)))
    return itersolver
 
itersolver = load_at(target)

# %% 
datalist = [pwlstats.solver_summary(solver) for solver in itersolver]

key = 'pwlm'
def get(key, datalist):
    return [(100*iteration, data[key]) for iteration, data in enumerate(datalist)]
    
def getT(key, datalist):
    return zip(*get(key, datalist))
get(key, datalist)


# %% 
# next test data is at 
# target = join(pili.root, "notebook/mdlpartition/")
# target = join(pwlstats.root, "run/partprio/_candidate_pwl/")
target = join(pwlstats.root, "run/partprio/trial/_top_0040/")


itersolver = load_at(target)
datalist = [pwlstats.solver_summary(solver) for solver in itersolver]
list(datalist[0].keys())


def plot_m(ax):
    data = get('pwlm', datalist)
    x,y = zip(*data)
    ax.plot(x, y)
    ax.set_xlabel('iteration')
    ax.set_ylabel('M')

def plot_loss(ax):
    data = get('loss', datalist)
    x,y = zip(*data)
    ax.plot(x, y)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')

def plot_dl(ax):
    data = get('dl', datalist)
    x,y = zip(*data)
    dl, m = zip(*y)
    # print(dl)
    # print(m)
    ax.plot(x, y, label=['DL', 'M'])
    ax.set_xlabel('iteration')
    ax.set_ylabel('dL')
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(8,4))
ax0, ax1, = axes
# ! the description length numbers reported here are a little bit confusing
plot_dl(ax0)
plot_loss(ax1)

# %% 
datalist[0]

# %% 
x, y = getT('loss', datalist)
y


np.gradient(y)