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
# A notebook for analysing our target simulations
#

# %% 
import warnings
import sys, os
import copy
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

import parameters
import stats
import command
import twutils
import eventanalyse
import readtrack
import collections

# %% 
# current simulation target
simtarget = "/home/dan/usb_twitching/run/825bd8f/target/t0"
with command.chdir(simtarget):
    ltarget = stats.load()
    args = parameters.thisread()
    ptrs = readtrack.eventset()
    mdtrs = readtrack.mdeventset()
_simobjective = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
simref = {name : twutils.make_get(name)(ltarget) for name in _simobjective}
_interest = ['dwell_time', 'k_spawn', 'pilivar',  'anchor_angle_smoothing_fraction']
# print(parameters.describe(args, target=_interest))
simpar = {par : args.pget(par) for par in _interest}
simpar['anchor_angle_smoothing_fraction'] *= np.pi/2
print('target has parameters', simpar)

# %% 
# 
ptr = ptrs[0]
pdata = eventanalyse._reorganize(ptr)
np.stack(pdata[0])['trigger']

# %% 
# mdevent analyse
md = mdtrs[0]
mddf = eventanalyse.pilistate(md)
mddf["deltat"] = mddf["last_retraction"] - mddf["first_retraction"]
dl = 0.004  # hardcoded displacement step
mddf["contraction"] = mddf["n_retractions"] * dl
mddf
# %% 
def describe_md(mddf):
    _interest_central_tendency = ["n_retractions",  "xydisplacement", "deltat", "contraction"]
    for par in _interest_central_tendency:
        _med = np.nanmedian(mddf[par])
        print("median(%s) = %s" % (par, str(_med)))
describe_md(mddf)
# check the number of pili that bind multiple times
collections.Counter(mddf["n_bindings"])
# what portion of the displacements are larger than the linearisation distance?
onestep = mddf["xydisplacement"] > 0.12
print("disp > 0.12: {}/{}".format( int(np.sum(onestep)), len(mddf) ))
# how does the proportion of step-sized displacements compare to the proportion of t=0.1 steps in fanjin.top?
# fanjin.py noteboo
# sns.histplot(mddf["xydisplacement"][onestep])
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(mddf["deltat"], ax=ax)
sns.histplot(mddf["deltat"][onestep], color='red', ax=ax)
# to make progress analysing "fast modes" (displacement greater than 0.12) we need to 

# %% 
plt.figure()
sns.histplot(mddf["n_retractions"])
plt.figure()
sns.histplot(mddf["xydisplacement"])
plt.figure()
ax = sns.histplot(mddf["last_retraction"] - mddf["first_retraction"])
ax.set_xlabel("$\Delta  t$")

