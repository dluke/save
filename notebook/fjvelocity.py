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
# attempt to reproduce velocity histogram in FJ paper (PNAS/2011)


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
import fjanalysis
import pwlstats

from skimage.restoration import denoise_wavelet
from skimage.restoration import  estimate_sigma

# %% 
mplstyle = {"font.size": 20}
notename = "fjvelocity"
publish = False

# %% 
original = _fj.trackload_original([2924])[0]
candidate = _fj.trackload([2924])[0]
lincandidate = _fj.lintrackload([2924])[0]

# %% 
def match_size(arr1, arr2):
    N = min(arr1.size, arr2.size)
    # explain 
    diff = max(arr1.size,arr2.size) - N
    if diff > 0:
        print("cut {} data points".format(diff))
    # ~ 
    return arr1[:N], arr2[:N]

# lvel, trail_lvel = original.get_speed(), original.get_speed(trail=True)
# lvel, trail_lvel = lincandidate.get_speed(), lincandidate.get_speed(trail=True)
# lvel, trail_lvel = candidate.get_speed(), candidate.get_speed(trail=True)

# lvel, trail_lvel = lincandidate.get_step_speed(), lincandidate.get_step_speed(trail=True)
# lvel, trail_lvel = match_size(lvel, trail_lvel)

# * instead of lineraising leading and trailing pole independently, use leading pole step_idx
def get_track_lvel(track):
    time = track["time"]
    s_time = time[track.step_idx]
    dt = s_time[1:] - s_time[:-1]
    x, y = track['x'], track['y']
    trail_x, trail_y = track['trail_x'], track['trail_y']
    s_x, s_y = x[track.step_idx], y[track.step_idx]
    s_trail_x, s_trail_y = trail_x[track.step_idx], trail_y[track.step_idx]
    xy = np.stack([s_x, s_y], axis=1)
    trail_xy = np.stack([s_trail_x, s_trail_y], axis=1)
    dx = xy[1:] - xy[:-1]
    trail_dx = trail_xy[1:] - trail_xy[:-1]
    lvel = norm(dx, axis=1)/dt
    trail_lvel = norm(trail_dx, axis=1)/dt
    return lvel, trail_lvel

def get_track_lvel(track):
    return track.get_speed(), track.get_speed(trail=True)

lvel, trail_lvel = get_track_lvel(lincandidate)

print(lvel.size, trail_lvel.size)

# def plot_lvel_histo
def plot_hist2d(ax, lvel, trail_lvel, log_scale=False):
    df = pd.DataFrame({'lead':lvel, 'trail':trail_lvel})
    sns.histplot(data=df, x='lead', y='trail', ax=ax, log_scale=log_scale, bins=20)
    ax.set_aspect("equal")
    lim = (0,32)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.plot(lim,lim, c='k', linestyle='--', alpha=0.5)

fig,ax= plt.subplots(figsize=(5,5))
plot_hist2d(ax, lvel, trail_lvel, log_scale=True)

# %% 
# * LOAD top data
top_idx = _fj.load_subset_idx()["top"]
top = _fj.load_subsets()["top"]

lveldata = [get_track_lvel(tr) for tr in top]
lead_data, trail_data = zip(*lveldata)
lvel, trail_lvel = np.concatenate(lead_data), np.concatenate(trail_data)

fig,ax= plt.subplots(figsize=(5,5))
plot_hist2d(ax, lvel, trail_lvel, log_scale=True)

# %% 
# * individual trajectories
# for i in range(5):
for i in range(50):
    lvel, trail_lvel  = get_track_lvel(top[i])
    fig, ax = plt.subplots(figsize=(5,5))
    plot_hist2d(ax, lvel, trail_lvel, log_scale=True)

# %% 
# lvel = lincandidate.get_step_speed()
lvel = lincandidate.get_speed()
lvel = candidate.get_speed()
lvel = original.get_speed()

lvel.min(), lvel.max()
lvel[lvel==0] = np.nan
sns.histplot(lvel, log_scale=(True,False))




