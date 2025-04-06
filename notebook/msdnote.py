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
# Add traditional kMSD as summary statistic 
# how long of a trajectory do we need to compute that and does FanJin data support it?

# %% 
import sys, os
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pili
import rtw
import _fj
import plotutils
import collections
import scipy.stats
import twanalyse
import pandas as pd
import parameters
import seaborn as sns

# %% 
subsets = _fj.load_subsets()
top = np.array(subsets["top"], dtype=object)
walking = np.array(subsets["walking"], dtype=object)

# %% 
import collections
Kmsddata = collections.namedtuple('Kmsddata', ['good', 'kmsd', 'mean', 'var'])

def condition(tr): # condition for computing   msd
    dx = tr.get_step_dx()
    dist = np.sqrt(np.sum(dx**2, axis=1))
    cd = np.sum(dist) > 10.0
    return cd
def kmsd(tr):
    p, cov, scaling, msd_n = twanalyse.kmsd_one(tr)
    kmsd, _intercept = p
    return kmsd
def summary(subset):
    good = np.array([condition(tr) for tr in subset])
    subset_kmsd = np.array([kmsd(tr) for tr in subset[good]])
    duration = np.array([tr.size for tr in subset[good]])
    mean, var = twutils.wmeanvar(subset_kmsd, ws=duration)
    return Kmsddata(good, subset_kmsd, mean, var)
# %%
wtup = summary(walking)
ttup = summary(top)
# %%
print('{}/{}'.format(int(np.sum(wtup.good)), len(top))) 
print('{}/{}'.format(int(np.sum(ttup.good)), len(top))) 

# %%
fig, ax = plt.subplots(figsize=(5,5))
shstyle = {"stat":"density"}
wgood, walking_kmsd, mean, var = wtup
print('walking')
print(mean, np.sqrt(var))
tgood, crawling_kmsd, mean, var = ttup
print('top')
print(mean, np.sqrt(var))
sns.histplot(walking_kmsd, label='walking', **shstyle)
sns.histplot(crawling_kmsd, color='orangered', label='crawling', **shstyle)
ax.set_xlabel('kMSD')
ax.legend()
# %% [markdown]
# the massive width in the kmsd of the walking trajectories
# indicates that their behvaiour (on the timescale of the experiment) is so varied
# as to make summary statistics over the whole set questionable
# we could try to use individual walking trajectories as reference data for ABC
# %%
# what about simulated crawling and walking?
# start with targets
t0 = twanalyse.load_simulation_target()
t3 = twanalyse.load_walking_target()

# %%
t0_kmsd = np.mean(np.array([kmsd(tr) for tr in t0]))
t3_kmsd = np.mean(np.array([kmsd(tr) for tr in t3]))
print('walking target kmsd', t3_kmsd)
print('crawling target kmsd', t0_kmsd)
# %%
# in order to examine the kmsd of the model more generally we need to add it as a summary statistic
# ...

# %%
# test lsummary
for subset, data in subsets.items():
    ld = fjanalysis.lsummary(data)
    print(subset, ld['kmsd']['mean'], ld['kmsd']['var'])
