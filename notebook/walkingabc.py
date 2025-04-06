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
# for walking we don't know how to reduce our problem to 3 dimensions
# but we want to know approximately what the posterior is in 4d anyway
# maybe we will go on to try ABC in 4 dimensions anyway

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

import readtrack
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
from abcimplement import rejection_abc

notename = "walkingabc"


# %%
plotting = False

# %%
notedir = os.getcwd()
root = pili.root
# candidate to compare against
plt.rcParams.update({
    'text.usetex': False,
    'axes.labelsize': 20,
    })
    
# %%
all_idx, ltrs = _fj.slicehelper.load_linearized_trs("all")
reference_idx = _fj.load_subset_idx()
# %%
objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate',
    'double_ltrs.qhat.estimate', 'double_ltrs.ahat.estimate']
refdf = fjanalysis.compute_reference_data(ltrs, reference_idx, objectives)
# %%
subset = "walking"
reference = refdf.iloc[4]
reference

# %%
# mc4dw = {}
# mc4dw["simdir"] = join(root, "../run/5bfc8b9/cluster/mc4d_walking")
# mc4dw["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'fanjin.walking.ks_statistic']
# mc4dw = abcimplement.load_problem_simulation(mc4dw)
# # %%
# # print problem
# print(mc4dw["problem"])
# nsamples = int(1e4)
# N = 200 
# print("accept {}/{}".format(N,nsamples))

# # %%
# # one statistic at a time
# mc4dw["params"] = mc4dw["data"].paramsdf(mc4dw["objectives"])
# statdf, statref = abcimplement.regularise_stats(mc4dw["params"], reference, mc4dw["objectives"])
# for objective in mc4dw["objectives"]:
#     _regdf = statdf[mc4dw["problem"]["names"] + [objective]]
#     _accepted = rejection_abc(_regdf, [objective], statref, N)
#     # rename = {k:k for k in _accepted.keys()}
#     # rename["anchor_angle_smoothing_fraction"] = "anchor"
#     # _accepted.rename(columns=rename, inplace=True)
#     abcimplement.problemplot4d(mc4dw["problem"], _accepted, objective)
#     plt.tight_layout()

# %% [markdown]
# lvel.mean sets good limits on dwell_time, unlike in crawling case !
# * why must anchor parameter be large? surely its less important for walking
#   - does the disagreement in anchor parameters imply that the slow crawling is really due to surface interaction?
# Can we interpret anything from ks_statistics?

# to start answering  these questions lets run again with a condition of atleast 1000 linear steps
# and some wider bounds (also velocity threshold on q/a estimator is in place)

# %%
new4dw = {}
new4dw["simdir"] = join(root, "../run/825bd8f/cluster/mc4d_walking")
new4dw["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'fanjin.walking.ks_statistic',
    'double_ltrs.qhat.estimate', 'double_ltrs.ahat.estimate']
    # 'quad_ltrs.qhat.estimate', 'quad_ltrs.ahat.estimate']
    # 'cell_ltrs.qhat.estimate', 'cell_ltrs.ahat.estimate']
new4dw = abcimplement.load_problem_simulation(new4dw)

# %%
# print problem
print(new4dw["problem"])
nsamples = int(1e4)
N = 200 
print("accept {}/{}".format(N,nsamples))

# %%
# one statistic at a time
new4dw["params"] = new4dw["data"].paramsdf(new4dw["objectives"])
new4dw["params"]
# %%
statdf, statref = abcimplement.regularise_stats(new4dw["params"], reference, new4dw["objectives"])
# statdf, statref = new4dw["params"], reference
# %%
accept = {}
for objective in new4dw["objectives"]:
    _regdf = statdf[new4dw["problem"]["names"] + [objective]]
    _accepted = rejection_abc(_regdf, [objective], statref, N)
    accept[objective] = _accepted

# %%
# if plotting:
if True:
    for objective, _accepted in accept.items():
        abcimplement.problemplot4d(new4dw["problem"], _accepted, objective)
        plt.tight_layout()

# %%
# find me a simulation with low dwell time and matching lvel.mean
lvel_acc = accept["lvel.mean"]
_s = lvel_acc.sort_values("dwell_time")
_s

new4dw["params"].iloc[685]

# %%
# _objectives = ["lvel.mean", "deviation.var", "ahat.estimate"]
_objectives = ["lvel.mean", "qhat.estimate", "ahat.estimate"]
_regdf = statdf[new4dw["problem"]["names"] + _objectives]
_accepted = rejection_abc(_regdf, _objectives, statref, N)
prime_accepted  = _accepted
# %%
print("walking reference summary statistics")
reference
# simref 
# {'lvel.mean': 0.17998735051859602, 
# 'deviation.var': 3.104551070742158, 
# 'qhat.estimate': 0.4580356484889145, 
# ahat.estimate': 0.4874124600988712}
# %%
m_par, v_par = abcimplement.mean_accepted(new4dw["problem"], _accepted)
abcimplement.describe_abc(new4dw, _accepted)

# %%
fig, axes = abcimplement.perfectplot4d(new4dw["problem"], _accepted, mpar=m_par)
pub.save_figure("walking_rejectionabc", notename)

# %% difference objectives
# tmp_objectives = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"]
tmp_objectives = ["lvel.mean", "deviation.var", "qhat.estimate"]
tmp_accepted = rejection_abc(statdf, tmp_objectives, statref, N)
m_par, v_par = abcimplement.mean_accepted(new4dw["problem"], tmp_accepted)
print(m_par)
if plotting:
    fig, axes = abcimplement.perfectplot4d(new4dw["problem"], tmp_accepted, mpar=m_par)

# %%
# plot velocity distribution of the optimal sample
udir = join(new4dw["simdir"], new4dw["lookup"][0][_accepted.index[0]])
twutils.sync_directory(udir)
args = parameters.thisread(directory=udir)
_trs = readtrack.trackset(ddir=join(udir, 'data/'))
bestsim = [_fj.linearize(tr) for tr in _trs]

# %%
print(parameters.describe(args, target=new4dw["problem"]["names"]))
# %%
# t2 = twanalyse.load_walking_target()
# lt2 = [_fj.linearize(tr) for tr in t2]
# t2_vel = np.concatenate([ltr.get_step_speed() for ltr in lt2])

# %%
# print("score", _accepted.iloc[0]["score"])

with plt.rc_context({'text.usetex': True}):
    fig, ax = plt.subplots(figsize=(6,5))
    sim_vel = np.concatenate([ltr.get_step_speed() for ltr in bestsim])
    # xlim = (0, np.quantile(sim_vel, 0.98))
    xlim = (0, 1.8)
    ref_vel = _fj.load_subset_speed()["walking"]
    ks, p = scipy.stats.ks_2samp(ref_vel, sim_vel)
    print("ks statistic = {:0.3f}".format(ks))
    common = {"bins": 50, 'binrange':xlim}
    sns.histplot(ref_vel, ax=ax, stat="density", alpha=0.5, label="reference", **common)
    sns.histplot(sim_vel, ax=ax, stat="density", color="orangered", alpha=0.5, label="sim", **common)
    ax.set_xlim(xlim)
    ax.legend()
    ax.set_xlabel(r"step velocity $(\mu m s^{-1})$", fontsize=30)
    ax.set_ylabel("density", fontsize=30)
    ax.grid(False)   
plt.tight_layout()
pub.save_figure("best_sim_walking_lvel_distrib", notename)

# %%
df = abcimplement.tabulate_inference(new4dw["problem"], prime_accepted, "fanjin walking")
pub.pickle_df(df, "bayes_inference", notename)
