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
# Rejection ABC against fanjin.top and simulated target using vret as  variable instead of \alpha

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

import pili.publication as pub

# %% 
plotting = False

# %% 
# config
notename = 'walkingvretabc'
warnings.filterwarnings("ignore")
plt.rcParams.update({
    'text.usetex': False,
    })

# %% 
# load fanjin 
all_idx, ltrs = _fj.slicehelper.load_linearized_trs("all")
reference_idx = _fj.load_subset_idx()
# %% 
objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
refdf = fjanalysis.compute_reference_data(ltrs, reference_idx, objectives)
simref = refdf.loc[refdf['subset'] == "walking"]
simref
# %% 
sim4d = {}
sim4d["simdir"] = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d_vret_walking/"
sim4d["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
sim4d = abcimplement.load_problem_simulation(sim4d)
sim4d["problem"]
# %% 
N = 200 
print('{}/{}'.format( N, sim4d["M"]))

# %% 
_objectives = sim4d["objectives"]
sim4d["params"] = sim4d["data"].paramsdf(_objectives)
abcimplement.transform_vret_parameter_data(sim4d)
statdf, statref = abcimplement.regularise_stats(sim4d["params"], simref, _objectives)

# %% 
_objectives = ["lvel.mean", "qhat.estimate", "ahat.estimate"]
_accepted = abcimplement.rejection_abc(statdf, _objectives, statref, N)
m_par, v_par = abcimplement.mean_accepted(sim4d["problem"], _accepted)
abcimplement.describe_abc(sim4d, _accepted)
prime_accepted = _accepted
# %%
lpar = [r'$\tau_{\mathrm{dwell}}$', r'$\kappa$', r'$v_{\mathrm{ret}}$', r'$k_{\mathrm{spawn}}$']
fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, mpar=m_par, lpar=lpar)
pub.save_figure('walking_vretabc', notename, fig)

