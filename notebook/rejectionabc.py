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
# approximate bayesian computation simple rejection method
# doing this method first is useful to see what bounds we should use for the larger samples
# and also to resolve crucial issues with our analysis
# i.e. the convergence of summary statistics

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

import pili
import parameters
import _fj
import fjanalysis
import twanalyse
import rtw
import sobol
import abcimplement
from abcimplement import rejection_abc

# %%
notedir = os.getcwd()
notename = 'rejectionabc'
root = pili.root
# candidate to compare against
simdir = join(root, "../run/5bfc8b9/cluster/mc4d")
plt.rcParams.update({
    'text.usetex': False,
    'figure.figsize': (20,20),
    'axes.labelsize': 20
    })


# %%
# load three parameter dataset
# use a dictionary to keep global namespace clean
mc3d = {}
mc3d["simdir"] = "/home/dan/usb_twitching/run/5bfc8b9/cluster/mc3d_frozen"
# remove the activty metric from the pool
# how to combine lvel similarity with other scores? Can't use l2 norm
mc3d["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'fanjin.top.ks_statistic']
# mc3d["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate']
mc3d = abcimplement.load_problem_simulation(mc3d)
# reload mc4d data as well
mc4d = {}
mc4d["simdir"] = simdir
mc4d = abcimplement.load_problem_simulation(mc4d)

# %%
all_idx, ltrs = _fj.slicehelper.load_linearized_trs("all")
reference_idx = _fj.load_subset_idx()
objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
refdf = fjanalysis.compute_reference_data(ltrs, reference_idx, objectives)
subset = "top"
reference = refdf.iloc[1]
N = 400
reference
# %%
mc3d["params"] = mc3d["data"].paramsdf(mc3d["objectives"])
statdf, statref = abcimplement.regularise_stats(mc3d["params"], reference, mc3d["objectives"])

# objective = 'fanjin.top.ks_statistic'
objective = 'deviation.var'
_regdf = statdf[mc3d["problem"]["names"] + [objective]]
accepted = rejection_abc(_regdf, [objective], statref, N)
accepted
# %%
# 3d plotting as 3 pairs of projections
from abcimplement import plot_accepted_projection
for objective in mc3d["objectives"]:
    _regdf = statdf[mc3d["problem"]["names"] + [objective]]
    _accepted = rejection_abc(_regdf, [objective], statref, N)
    fig, axes = plot_accepted_projection(mc3d["problem"], _accepted)
    fig.suptitle(objective)

# %%
# all three remaining simple metrics
_objectives = ["lvel.mean", "deviation.var"]
# _objectives = ["deviation.var", "fanjin.top.ks_statistic"]
_regdf = statdf[mc3d["problem"]["names"] + _objectives]
_accepted = rejection_abc(_regdf, _objectives, statref, N)
fig, axes = plot_accepted_projection(mc3d["problem"], _accepted)
fig.suptitle(str(_objectives))

# %%
# lookup the step counts for this whole dataset
nsteps = mc3d["data"].get("linearsteps")
qhat = mc3d["data"].get("qhat.estimate")
qhat_err = mc3d["data"].get("qhat.err")
qdata = pd.DataFrame({"nsteps": nsteps, "q": qhat, "qhat_err": qhat_err})
_nsteps = mc4d["data"].get("linearsteps")
ahat = mc4d["data"].get("ahat.estimate")
ahat_err = mc4d["data"].get("ahat.err")
adata = pd.DataFrame({"nsteps": _nsteps, "a": ahat , "ahat_err": ahat_err})

# %%
# lookup the step counts for this whole dataset
plt.rcParams.update({'text.usetex': False})
fig, axes = plt.subplots(1,2,figsize=(10,5))
g = sns.scatterplot(data=qdata,
    x="nsteps", y="qhat_err", ax=axes[0]) 
axes[0].set_ylim((0,1))
# g.ax.axhline(0.05)
g = sns.scatterplot(data=adata,
    x="nsteps", y="ahat_err", ax=axes[1]) 
# g.ax.set_xlim((0,1))
fig.suptitle("convergence test")

# %% [markdown]
# * plot this for all four objectives 
#   - don't have a standard error estimate for deviation.var, ahat is in another dataset
#     that leaves lvel.mean and qhat, lvel.mean should converge faster 
#     (also its error is currently calculated from correlated samples)
# * map low step count back to parameter space
# what err should we aim for? 0.05?
# * (probably) implement adaptive simulation max time
# %%
# map nsteps < 500 back onto parameter axes
_paramdf  = mc3d["data"].paramsdf(["linearsteps"])
_df3d = _paramdf[_paramdf["linearsteps"] < 500]
print("selected {}/{}".format(len(_df3d) , len(_paramdf)))
plt.rcParams.update({'axes.labelsize': 20})
abcimplement.problemplot3d(mc3d["problem"], _df3d, hue="linearsteps", snskw={"legend":False})
# nsteps varies mainly with k_spawn but simulations with < 500 steps span the entire space
# we should check if these runs actually failed 

# %%
# how do we do against simulated data?
# we need to update local.json with something like simulated.<uid>.ks_statistic
# or else it would be useful to store lvel locally ...

# %% [markdown]
# our summary statistics are not informing us as to the value of dwell_time 
# is this because the forward motion is constrained by anchor parameter?
# we can plug in tala estimates of 1.0s, 2.5s and report estimates for pilivar and k_spawn
# approximately (pilivar = 2.5, k_spawn = 5.0)
# 

# %%
# now the same but with all pairs (4 choose 2) of metrics

#