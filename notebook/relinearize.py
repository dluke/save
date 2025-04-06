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
# adjust the linearization distance for the walking reference simulation / experimental data and see where the third peak vanishes

# %%  
import warnings
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
import command
import stats
import pili
import parameters
import _fj
import fjanalysis
import twanalyse
import rtw
import sobol
import abcimplement
import twutils
# %%  
t3 = twanalyse.load_walking_target()
ltlookup = {}
names = ['lin', 'double', 'quad','cell']
dsteps = [0.12, 0.24, 0.48, 1.0]
namedata = {name: dstep for name,  dstep in zip(names, dsteps)}
for name, dstep in zip(names, dsteps):
    ltlookup[name] = [_fj.linearize(tr, step_d=dstep) for tr in t3]
ltlookup
# %%  
for name in ltlookup:
    _ltr =  ltlookup[name]
    lvel = np.concatenate([tr.get_step_speed() for tr in _ltr])
    fig, ax = plt.subplots(figsize=(5,5))
    xlim = [0,2]
    sns.histplot(lvel, binrange=xlim, ax=ax)
    ax.set_title('dstep = {}'.format(namedata[name]))

# %%  
# ----------------------------------------------------------------
# walking
# ----------------------------------------------------------------
walking = _fj.load_subsets()['walking']
# %%  
wltlookup = {}
for name, dstep in zip(names, dsteps):
    wltlookup[name] = [_fj.linearize(tr, step_d=dstep) for tr in walking]
# %%  
for name in wltlookup:
    _ltr =  wltlookup[name]
    lvel = np.concatenate([tr.get_step_speed() for tr in _ltr])
    fig, ax = plt.subplots(figsize=(5,5))
    xlim = [0,2]
    sns.histplot(lvel, binrange=xlim, ax=ax)
    ax.set_title('dstep = {}'.format(namedata[name]))



# %%  