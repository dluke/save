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
# mdlvelocity.py is getting out of hand.
# we want build more intuition by looking at some simulated velocity distributions


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

import readtrack
import stats
import command
import twanalyse

# %% 
mplstyle = {"font.size": 20}
notename = "fitsimvel"
publish = False

# %% 
rundir = join(pili.root, '../run')
path = join(rundir, "825bd8f/target/t0/")
t0 = readtrack.trackset(ddir=join(path, 'data/'))
with command.chdir(path):
    local = stats.load()
    lvel = np.load(join(path,"lvel.npy"))
    
# %% 
sns.histplot(lvel)
print("ntaut", local["ntaut"]["mean"])

# %% 
vel = np.concatenate([tr.get_speed() for tr in t0])
zero_idx = vel == 0
def describe_ratio(a, b):
    print("zeros {}/{} ({:.3f})".format(a, b, a/b))
describe_ratio(np.sum(zero_idx), vel.size)
nonzero = vel[~zero_idx]
sns.histplot(nonzero, bins=50)

# %%
# pull an acceped sample with larger ntaut.mean
path = join(rundir, "825bd8f/cluster/mc4d/_u_DbaHFe2d")
acc = readtrack.trackset(ddir=join(path, 'data/'))

dx = 0.1 * np.concatenate([tr.get_speed() for tr in acc])
zero_idx = dx == 0
def describe_ratio(a, b):
    print("zeros {}/{} ({:.3f})".format(a, b, a/b))
describe_ratio(np.sum(zero_idx), dx.size)
nonzero = dx[~zero_idx]

fig, ax = plt.subplots(figsize=(6,4))
shstyle = dict(element="step", stat="density", fill=False, alpha=0.8)
sns.histplot(nonzero, bins=50, ax=ax, **shstyle )

def make_error(sigma):
    def error(x):
        return scipy.stats.norm(0, sigma).pdf(x)
    return error

sigma = 0.0126

def mirror_data(arr):
    return np.concatenate([arr, -arr])

import statsmodels.api as sm
kde = sm.nonparametric.KDEUnivariate(mirror_data(nonzero))
kde.fit()
kde.fit(bw=kde.bw/2)
zero_cut_idx = np.searchsorted(kde.support, 0)
support, density = kde.support[zero_cut_idx:], 2 * kde.density[zero_cut_idx:]

error = make_error(sigma)
kde_delta = np.diff(kde.support)[0]
err_support = np.arange(-3*sigma, 3*sigma, kde_delta)



# regularise the support
support = np.linspace(err_support[0], support[-1], 4000)
density = 2 * kde.evaluate(support)
density[:np.searchsorted(support, 0)] = 0
err_support = support[0:np.searchsorted(support, err_support[-1])]
error_density = error(err_support)


ax.plot(support, density, label="kde")
ax.plot(err_support, error_density, label="err")
ax.axvline(0, color='k', alpha=0.2)


conv_density = scipy.signal.convolve(density, error_density, mode="same")
conv_density /= scipy.integrate.simpson(conv_density, support)
print(support.size, err_support.size, conv_density.size)

ax.plot(support, conv_density, label="convolution")

xlim = (-0.08,0.12)
ax.set_xlim(xlim)

ax.legend()


# %%
# TODO: same thing but for mapped velocity (fast PWL solve on simulation)
