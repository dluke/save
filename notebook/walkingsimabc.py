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
notename = 'walkingsimabc'

# %% 
# config
plt.rcParams.update({
    'text.usetex': False,
    'figure.figsize': (20,20),
    'axes.labelsize': 16
    })
# %% 
# configuration word
plotting = True

# %% 
simtarget = "/home/dan/usb_twitching/run/825bd8f/target/t3"
with command.chdir(simtarget):
    ltarget = stats.load()
    args = parameters.thisread()
_simobjective = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate','kmsd.mean',
    'double_ltrs.qhat.estimate', 'double_ltrs.ahat.estimate']
simref = {name : twutils.make_get(name)(ltarget) for name in _simobjective}
print("simref", simref)
_interest = ['dwell_time', 'k_spawn', 'pilivar',  'anchor_angle_smoothing_fraction']
# print(parameters.describe(args, target=_interest))
simpar = {par : args.pget(par) for par in _interest}
simpar['anchor_angle_smoothing_fraction'] *= np.pi/2
simpar
# %% 
sim4d = {}
sim4d["simdir"] = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d_walking"
sim4d["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'fanjin.walking.ks_statistic',
    'double_ltrs.qhat.estimate', 'double_ltrs.ahat.estimate']
sim4d = abcimplement.load_problem_simulation(sim4d)
sim4d["problem"]

# %% 
N = 200 
print('{}/{}'.format( N, sim4d["M"]))
# %% 
_objectives = _simobjective 
sim4d["params"] = sim4d["data"].paramsdf(_objectives)
abcimplement.transform_anchor_parameter_data(sim4d)
statdf, statref = abcimplement.regularise_stats(sim4d["params"], simref, _objectives)
#
# %%
if plotting:
    for objective in _objectives:
        _accepted = abcimplement.rejection_abc(statdf, [objective], statref, N)
        fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=simpar)
        fig.suptitle(objective)

# %%
# all three simple metrics
warnings.filterwarnings("ignore")
_objectives = ["lvel.mean", "qhat.estimate", "ahat.estimate"]
_regdf = statdf[sim4d["problem"]["names"] + _objectives]
_accepted = abcimplement.rejection_abc(_regdf, _objectives, statref, N)
m_par, v_par = abcimplement.mean_accepted(sim4d["problem"], _accepted)
abcimplement.describe_abc(sim4d, _accepted)
prime_accepted = _accepted
# %%
fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=simpar, mpar=m_par)
pub.save_figure('walking_simabc', notename, fig)

# %%
df = abcimplement.tabulate_inference(sim4d["problem"], prime_accepted, "simulated t3")
pub.pickle_df(df, "bayes_inference_t3", notename)

# %%
walking = _fj.load_subsets()['walking']
fjld = fjanalysis.lsummary(walking)
# %%
# compare kmsd
params = sim4d["params"]
_accdf = params.iloc[_accepted.index]
acc_kmsd = _accdf['kmsd.mean']
print(np.mean(acc_kmsd), np.std(acc_kmsd))
# compare with fanjin
fjld['kmsd']
# we can write this up if we want ~ 

# %% 
# what if we only use lvel.mean and deviation.var
def load_target_at(dir):
    with command.chdir(dir):
        ltarget = stats.load()
        args = parameters.thisread()
    _simobjective = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
    simref = {name : twutils.make_get(name)(ltarget) for name in _simobjective}
    interest = ['dwell_time', 'k_spawn', 'pilivar',  'anchor_angle_smoothing_fraction']
    simpar = {par : args.pget(par) for par in interest}
    if 'anchor_angle_smoothing_fraction' in simpar:
        simpar['anchor_angle_smoothing_fraction'] *= np.pi/2
    return simpar, simref
# _simpar, _simref = load_target_at("/home/dan/usb_twitching/run/825bd8f/target/t3")

# _objectives = ["lvel.mean", "deviation.var"]
# _statdf, _statref = abcimplement.regularise_stats(sim4d["params"], _simref, _objectives)
# _accepted = abcimplement.rejection_abc(_statdf, _objectives, _statref, N)
# m_par, v_par = abcimplement.mean_accepted(sim4d["problem"], _accepted)
# abcimplement.describe_abc(sim4d, _accepted)
# if plotting:
#     fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=_simpar, mpar=m_par)

# %% [markdown]
# ##################################### 
# ##################################### 
# change target t2

# %% 
simtarget = "/home/dan/usb_twitching/run/825bd8f/target/t2"
_simpar, _simref = load_target_at(simtarget)
_simref

# %% 
# repeat for this simulation target
_objectives = ["lvel.mean", "qhat.estimate", "ahat.estimate"]
_statdf, _statref = abcimplement.regularise_stats(sim4d["params"], _simref, _objectives)
_accepted = abcimplement.rejection_abc(_statdf, _objectives, _statref, N)
m_par, v_par = abcimplement.mean_accepted(sim4d["problem"], _accepted)
abcimplement.describe_abc(sim4d, _accepted)

# %% 
fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=_simpar, mpar=m_par)
pub.save_figure('walking_simabc_t2', notename, fig)

# %% 
df = abcimplement.tabulate_inference(sim4d["problem"], _accepted, "simulated t2", simpar=_simpar)
pub.pickle_df(df, "bayes_inference_t2", notename)
df

# %% [markdown]
# ##################################### 
# ##################################### 
# change target
# %% 
simtarget = "/home/dan/usb_twitching/run/825bd8f/target/t4"
_simpar, _simref = load_target_at(simtarget)

# %% 
# repeat for this simulation target
_objectives = ["lvel.mean", "qhat.estimate", "ahat.estimate"]
_statdf, _statref = abcimplement.regularise_stats(sim4d["params"], _simref, _objectives)
_accepted = abcimplement.rejection_abc(_statdf, _objectives, _statref, N)
m_par, v_par = abcimplement.mean_accepted(sim4d["problem"], _accepted)
abcimplement.describe_abc(sim4d, _accepted)

# %% 
fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=_simpar, mpar=m_par)
pub.save_figure('walking_simabc_t4', notename, fig)

# %% 
debug = False
if debug:
    dens = abcimplement._statsuniformkde(_accepted, "dwell_time")
    print('bandwidth', dens.bw)
    fig, ax = plt.subplots(figsize=(5,5))
    sns.histplot(_accepted["dwell_time"], ax=ax, stat="density")
    low, high = sim4d["problem"]["bounds"][0]
    space = np.linspace(low, high, 1000)
    ax.plot(space, dens.pdf(space), c='k')

# %% 
df = abcimplement.tabulate_inference(sim4d["problem"], _accepted, "simulated t4", simpar=_simpar)
pub.pickle_df(df, "bayes_inference_t4", notename)
df

# %% 
# what if we only use lvel.mean and deviation.var
_objectives = ["lvel.mean", "deviation.var"]
_statdf, _statref = abcimplement.regularise_stats(sim4d["params"], _simref, _objectives)
_accepted = abcimplement.rejection_abc(_statdf, _objectives, _statref, N)
m_par, v_par = abcimplement.mean_accepted(sim4d["problem"], _accepted)
abcimplement.describe_abc(sim4d, _accepted)
if plotting:
    fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=_simpar, mpar=m_par)
