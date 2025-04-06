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
import pandas as pd
import seaborn as sns
import scipy.stats

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
# current simulation target
simtarget = "/home/dan/usb_twitching/run/825bd8f/target/t0"
with command.chdir(simtarget):
    ltarget = stats.load()
    args = parameters.thisread()
_simobjective = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
simref = {name : twutils.make_get(name)(ltarget) for name in _simobjective}
_interest = ['dwell_time', 'k_spawn', 'pilivar',  'anchor_angle_smoothing_fraction']
# print(parameters.describe(args, target=_interest))
simpar = {par : args.pget(par) for par in _interest}
simpar['anchor_angle_smoothing_fraction'] *= np.pi/2
print('target has parameters', simpar)

# %% 
def describe_pili(lt):
    _interest = ["lifetime", "bound_time", "taut_time", "effective_contract_length", 
        "bound_pili_participation", "taut_pili_participation", 
        "nbound.mean", "ntaut.mean", "l_total", "npili.mean"]
    for name in _interest:
        print(name, twutils.make_get(name)(lt))

describe_pili(ltarget)



# %% 
# obtain the effective retraction behaviour

