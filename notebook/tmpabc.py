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
# Rejection ABC against a simulated target
# TMP FILE copied from simabc.py
# DO NOT EDIT

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
plt.rcParams.update({
    'text.usetex': False,
    'figure.figsize': (20,20),
    'axes.labelsize': 16
    })
notename = 'simabc'
verbose = False


# %% 
simtarget = "/home/dan/usb_twitching/run/825bd8f/target/t0"
with command.chdir(simtarget):
    ltarget = stats.load()
    args = parameters.thisread()
_simobjective = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'kmsd.mean',
    'double_ltrs.qhat.estimate', 'double_ltrs.ahat.estimate', 
    'quad_ltrs.qhat.estimate', 'quad_ltrs.ahat.estimate', 
    'cell_ltrs.qhat.estimate', 'cell_ltrs.ahat.estimate']
simref = {name : twutils.make_get(name)(ltarget) for name in _simobjective}
_interest = ['dwell_time', 'k_spawn', 'pilivar',  'anchor_angle_smoothing_fraction']
# print(parameters.describe(args, target=_interest))
simpar = {par : args.pget(par) for par in _interest}
simpar['anchor_angle_smoothing_fraction'] *= np.pi/2
simpar, simref
# %% 
sim4d = {}
sim4d["simdir"] = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d"
sim4d["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'fanjin.top.ks_statistic', 'kmsd.mean',
    'double_ltrs.qhat.estimate', 'double_ltrs.ahat.estimate', 
    'quad_ltrs.qhat.estimate', 'quad_ltrs.ahat.estimate', 
    'cell_ltrs.qhat.estimate', 'cell_ltrs.ahat.estimate']
sim4d = abcimplement.load_problem_simulation(sim4d)
sim4d["problem"]

# %% 
# ABC config
N = 200 
print('{}/{}'.format( N, sim4d["M"]))

# %%
# one statistic at a time
_objectives = _simobjective 
sim4d["params"] = sim4d["data"].paramsdf(_objectives)
abcimplement.transform_anchor_parameter_data(sim4d)
statdf, statref = abcimplement.regularise_stats(sim4d["params"], simref, _objectives)
#
statref
# %%
special_stat = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
spar = [r'$\langle u \rangle$', r'$Var(\theta_D)$', r'$\hat{q}$', r'$\hat{a}$']
_pretty = dict([(a,b) for a, b in zip(special_stat, spar)])
special = True
# TMP
_objectives = ['qhat.estimate', 'ahat.estimate',
    'double_ltrs.qhat.estimate', 'double_ltrs.ahat.estimate', 
    'quad_ltrs.qhat.estimate', 'quad_ltrs.ahat.estimate', 
    'cell_ltrs.qhat.estimate', 'cell_ltrs.ahat.estimate'
    ]
_titles = [
    'persistence, step = 0.12', 'activity, step = 0.12',
    'persistence, step = 0.24', 'activity, step = 0.24',
    'persistence, step = 0.48', 'activity, step = 0.48',
    'persistence, step = 1.00', 'activity, step = 1.00',
]
# ~TMP
if plotting or special:
    for i, objective in enumerate(_objectives):
        _regdf = statdf[sim4d["problem"]["names"] + [objective]]
        _accepted = abcimplement.rejection_abc(_regdf, [objective], statref, N)
        fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=simpar)
        fig.suptitle(_titles[i], fontsize=40)
        if objective in special_stat:
            pass
            plt.savefig('jure/sim_crawling_abc_statistic_{}.png'.format(objective))
            # fig.suptitle(_pretty[objective])
            # pub.save_figure('sim_crawling_abc_statistic_{}'.format(objective), notename, fig, config={"svg":False})

# %% [markdown]
# because we have easy access to this data, we list the mean displacment-per-pilus 
# of the accepted samples for approximate bayesian computation using persistence
# and activity statistics ALONE as the ABC statistics but with varying linearisation step sizes.

# %%
# FOR JURE
# compute mean per TFP displacement for each set of accepted samples
# _delta = sim4d["data"].get("effective_contract_length.mean")
_delta = sim4d["data"].get("pdisp.mean")
print('pdist.mean (all samples)', np.mean(_delta))
lst = []
for i, objective in enumerate(_objectives):
    _regdf = statdf[sim4d["problem"]["names"] + [objective]]
    _accepted = abcimplement.rejection_abc(_regdf, [objective], statref, N)
    acc_delta_l = _delta[_accepted.index]
    lst.append(np.mean(acc_delta_l))

r = [0.12,0.24,0.48,1.0]
a, b = lst[::2], lst[1::2]
df = pd.DataFrame({"step":r, "per TFP displacement (persistence)": a, "per TFP displacement (activity)" : b})
df

