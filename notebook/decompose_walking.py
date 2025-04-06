
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
# the velocity distribution of the walking trajectories has in interesting 
# shape. Looks like three shifted exponential decays.
# The the second peak should be free traction but what is the third peak?

# we of course will use simulated data to find out

# %% 
import sys, os
import copy
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import scipy.stats

import readtrack
import pili
import parameters
import _fj
import fjanalysis
import twanalyse
import rtw
import twutils

import pili.publication as pub

notename = "decompose_walking"
# %%
fig, ax = plt.subplots(figsize=(5,5))
t3 = twanalyse.load_walking_target()
lt3 = [_fj.linearize(tr) for tr in t3]
t3_vel = np.concatenate([ltr.get_step_speed() for ltr in lt3])
xlim = (0, np.quantile(t3_vel, 0.99))
eyebin = [0, 0.6, 1.2]
ax.axvline(0.60)
ax.axvline(1.2)
sns.histplot(t3_vel, binrange=xlim)


# %%
vel = t3_vel
base = (vel > eyebin[0]) & (vel < eyebin[1])
free = (vel > eyebin[1]) & (vel < eyebin[2])
fast = (vel > eyebin[2]) 

# %%
tr = lt3[0]
fig, ax = plt.subplots(figsize=(8,8))
for _i in range(tr.get_nsteps()):
    sp, su = tr.step_idx[_i], tr.step_idx[_i+1]
    x = tr.track['x']
    y = tr.track['y']
    color = 'k'
    if base[_i]:
        color = 'b'
    elif free[_i]:
        color = 'yellow'
    elif fast[_i]:
        color = 'red'

    ax.plot([x[sp],x[su]], [y[sp],y[su]], c=color)

# %%
# extract the statistics, nbound, ntaut (that's all we have in 0.1 second tracking file.)
vnames = ['base', 'free', 'fast']
intervals = ['pre', 'aft', 'interval']
statistic = ['nbound', 'ntaut']
import itertools
keys = ['.'.join(triple) for triple in list(itertools.product(vnames, intervals, statistic))]

# %%
def binning(vel):
    base = (vel > eyebin[0]) & (vel < eyebin[1])
    free = (vel > eyebin[1]) & (vel < eyebin[2])
    fast = (vel > eyebin[2]) 
    vbin = np.empty(len(vel), dtype=int)
    vbin[base] = 0
    vbin[free] = 1
    vbin[fast] = 2
    return vbin

def decompose_tracking_stats(tr):
    data = {_key:[] for _key in keys}
    nbound = tr['nbound']
    ntaut = tr['ntaut']
    vel = tr.get_step_speed()
    vbin = binning(vel)
    for _i in range(tr.get_nsteps()):
        sp, su = tr.step_idx[_i], tr.step_idx[_i+1]
        bin_name = vnames[vbin[_i]]
        interval_data = nbound[sp], nbound[su], np.mean(nbound[sp:su+1])
        for _k, interval_name in enumerate(intervals):
            key = '.'.join([bin_name, interval_name, 'nbound'])
            data[key].append( interval_data[_k])
        interval_data = ntaut[sp], ntaut[su], np.mean(ntaut[sp:su+1])
        for _k, interval_name in enumerate(intervals):
            key = '.'.join([bin_name, interval_name, 'ntaut'])
            data[key].append( interval_data[_k])
    return data

datas = [decompose_tracking_stats(tr) for tr in lt3]
# concatentate
data = {k : np.concatenate([d[k] for d in datas]) for k in keys}
# reduce 
reduced = {}
for k in data:
    reduced[k] = np.mean(data[k])
reduced
# split data by nbound/ntaut
reduced_nbound = {k: reduced[k] for k in reduced if 'nbound' in k.split('.')}
print(reduced_nbound)
# reduced_nbound
reduced_ntaut = {k: reduced[k] for k in reduced if 'ntaut' in k.split('.')}
reduced_ntaut





