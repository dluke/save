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
import pili.support as support
import parameters
import _fj
import fjanalysis
import twanalyse
import rtw
import sobol
import abcimplement
import twutils


import pili.publication as pub
import thesis.publication as thesis
from abcimplement import rejection_abc

notename = "walkingabc"


# %%
plotting = False
publish = True

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
objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
refdf = fjanalysis.compute_reference_data(ltrs, reference_idx, objectives)


# %%
subset = "walking"
reference = refdf.iloc[4]
reference

# %%
new4dw = {}
new4dw["simdir"] = join(root, "../run/825bd8f/cluster/mc4d_walking")
new4dw["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'fanjin.walking.ks_statistic']
new4dw = abcimplement.load_problem_simulation(new4dw)

# %%
# print problem
print(new4dw["problem"])
nsamples = int(1e4)
N = 50
print("accept {}/{}".format(N,nsamples))

# %%
# one statistic at a time
new4dw["params"] = new4dw["data"].paramsdf(new4dw["objectives"])
new4dw["params"]

# %%

abcimplement.transform_anchor_parameter_data(new4dw)
statdf, statref = abcimplement.regularise_stats(new4dw["params"], reference, new4dw["objectives"])
# statdf, statref = new4dw["params"], reference

# %%
accept = {}
for objective in new4dw["objectives"]:
    _regdf = statdf[new4dw["problem"]["names"] + [objective]]
    _accepted = rejection_abc(_regdf, [objective], statref, N)
    accept[objective] = _accepted

# %%
# _objectives = ["lvel.mean", "deviation.var", "ahat.estimate"]
# _objectives = ["lvel.mean", "qhat.estimate", "ahat.estimate"]

# _objectives = ["fanjin.walking.ks_statistic", "deviation.var", "qhat.estimate", "ahat.estimate"]


_objectives = ["fanjin.walking.ks_statistic", "deviation.var"]
_objectives = ["fanjin.walking.ks_statistic", "deviation.var", "qhat.estimate", "ahat.estimate"]

_objectives = ["lvel.mean", "deviation.var", "ahat.estimate"]
_objectives = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"]


N = 50
_regdf = statdf[["uid"] + new4dw["problem"]["names"] + _objectives]
_accepted = rejection_abc(_regdf, _objectives, statref, N)
prime_accepted  = _accepted

_accepted["score"].min()

# %%
# reference
_accepted.iloc[0]
udir = join(new4dw["simdir"], new4dw["lookup"][0][prime_accepted.index[0]])
udir

# _trs = readtrack.trackset(ddir=join(udir, 'data/'))
# bestsim = [_fj.linearize(tr) for tr in _trs]
# prime_accepted.iloc[0]


# %%
# reference
statref

# %%
print("walking reference summary statistics")
reference
# simref 
# {'lvel.mean': 0.17998735051859602, 
# 'deviation.var': 3.104551070742158, 
# 'qhat.estimate': 0.4580356484889145, 
# ahat.estimate': 0.4874124600988712}

# %%
problem = new4dw["problem"]
df = abcimplement.tabulate_inference(problem, _accepted, _objectives)
m_par = {k : v for k, v  in zip(df["parameter"], df["MLE"])}
df

# pub.save_dfhtml(df, "fj_walkingabc", notename)
# df

# %%
reference

# %%
fig, axes = abcimplement.perfectplot4d(new4dw["problem"], _accepted, mpar=m_par)
publish = True
if publish:
    pub.save_figure("walking_rejectionabc_t2", notename)
# pub.save_figure("walking_rejectionabc_ks_devvar", notename)


# %% 
# save the accepted samples
simdir = new4dw["simdir"]
def save_accepted_samples(simdir):
    accept_uid = _accepted["uid"].to_numpy()
    cachefile = join(simdir, "standard_accepted_uid.npy")
    print("saving accepted uid list to ", cachefile)
    np.savetxt(cachefile, accept_uid, fmt="%s")
save_accepted_samples(simdir)

# %%
# pull the xydisp.npy data for the accepted samples
simdir = new4dw["simdir"]
def load_accepted_uid(simdir):
    return np.loadtxt(join(simdir, "standard_accepted_uid.npy"), dtype=object)

def load_accepted_xydisp(simdir):
    uidlist = load_accepted_uid(simdir)
    data = [np.load(join(simdir, uid, 'xydisp.npy')) for uid in uidlist]
    data = list(map(lambda arr: arr[arr>0], data))
    return data

# load_accepted_uid(simdir)
uidlist = load_accepted_uid(simdir)
xydispdata = load_accepted_xydisp(simdir)

# %%
_accepted["anchor_angle_smoothing_fraction"].max()

# %%
median_pdisp = np.array([np.median(arr) for arr in  xydispdata])
median_pdisp
sns.histplot(median_pdisp)


# %%
mean_pdisp = np.array([np.mean(arr) for arr in  xydispdata])
sns.histplot(mean_pdisp)

# %%

# _accepted["uid"]
alpha = np.pi/2 * _accepted["anchor_angle_smoothing_fraction"]
fig, ax = plt.subplots(figsize=(6,4))
scatstyle = {'edgecolor':'white', 'linewidth' : 1.0, 'alpha' : 0.5, 's': 60}
ax.scatter(alpha, median_pdisp, **scatstyle)
ax.set_xlim(0, np.pi/2)


# sns.histplot(xydispdata[0])


# %%
# best accepted
_accepted.iloc[0:10]

# %%
# plot velocity distribution of the optimal sample
uid = new4dw["lookup"][0][_accepted.index[0]]
udir = join(new4dw["simdir"], new4dw["lookup"][0][_accepted.index[0]])
print('best uid', uid)
local = new4dw["lduid"][uid]
print('bound')
local['nbound']
print('taut')
local['ntaut']

# %%
udir = join(simdir, uid, "localrun")

args = parameters.thisread(directory=udir)
_trs = readtrack.trackset(ddir=join(udir, 'data/'))
best_trs = [_fj.linearize(tr) for tr in _trs]

# %%
import command
import pilush
with command.chdir(udir):
    ptdata = readtrack.piliset()[0]


# %%
visible_ranges = [(0,99.0), (0.3,99.0),(1.0,99.0),(2.0,99.0)]

timelime, npy, npy_in_range, bound_in_range, taut_in_range = pilush.count_visible(ptdata, visible_ranges=visible_ranges)


mean_npili = np.mean(npy)
form = 'Average number of pili in range {} is {:3.1f}/{:3.1f}'
for vr, nlist in list(npy_in_range.items()):
    print((form.format(vr, np.mean(nlist), mean_npili)))
print()

mean_npili = np.mean(bound_in_range[visible_ranges[0]])
for vr, nlist in list(bound_in_range.items()):
    print((form.format(vr, np.mean(nlist), mean_npili)))
print()

mean_npili = np.mean(taut_in_range[visible_ranges[0]])
for vr, nlist in list(taut_in_range.items()):
    print((form.format(vr, np.mean(nlist), mean_npili)))



# %%

visible_ranges = [(0,99.0)]
timelime, npy, npy_in_range, bound_in_range, taut_in_range = pilush.count_visible(ptdata, visible_ranges=visible_ranges)

# %%
# stacked bars

visible_ranges = [(0,99.0), (0.3,99.0),(1.0,99.0),(2.0,99.0)]

timelime, npy, npy_in_range, bound_in_range, taut_in_range = pilush.count_visible(ptdata, visible_ranges=visible_ranges)
clrange = [npy_in_range, bound_in_range, taut_in_range]

# %%

with mpl.rc_context({'font.size': 20, 'text.usetex': True}):
    fig, ax = plt.subplots(figsize=(3,4))
    # support.plot_stacked_bars(ax, clrange, visible_ranges)
    support.plot_bars(ax, clrange, visible_ranges)
    ax.locator_params(nbins=6, axis='y')

if True:
    pub.save_figure("walking_best_accepted_tfp_count_bars.png", notename)
    thesis.save_figure("walking_best_accepted_tfp_count_bars.png", notename)


# %%
# pull the best simulation
# twutils.sync_directory(udir)
args = parameters.thisread(directory=udir)
_trs = readtrack.trackset(ddir=join(udir, 'data/'))
bestsim = [_fj.linearize(tr) for tr in _trs]

# %%
# check the peak at 0.6 in lvel distribution by relinearizing the fanjin trajectories
ref_vel_idx = _fj.load_subset_idx()["walking"]
wavelet = _fj.trackload(ref_vel_idx)

dstep = 0.10
ltrs = [_fj.linearize(tr, step_d=dstep) for tr in wavelet]
ltrs

# %% 
re_ref_vel = np.concatenate([ltr.get_step_speed() for ltr in ltrs])
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(re_ref_vel)
xlim = (0, 1.8)
ax.set_xlim(xlim)
# ! the peak at 0.6/1.2 isn't real, just a detail of the coarse graining


# %%


with plt.rc_context({'text.usetex': True, 'font.size':24}):

    fig, ax = plt.subplots(figsize=(5.5,5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    sim_vel = np.concatenate([ltr.get_step_speed() for ltr in bestsim])
    # xlim = (0, np.quantile(sim_vel, 0.98))
    xlim = (0, 1.8)
    ref_vel = _fj.load_subset_speed()["walking"]
    ks, p = scipy.stats.ks_2samp(ref_vel, sim_vel)
    print("ks statistic = {:0.3f}".format(ks))
    hstyle = {'stat':'density', 'alpha':0.3, 'fill':True, 'element':'step'}
    common = {"bins": 50, 'binrange':xlim}
    common.update(hstyle)
    sns.histplot(ref_vel, ax=ax, label="reference", **common)
    sns.histplot(sim_vel, ax=ax, color="orangered", label="simulation", **common)
    ax.set_xlim(xlim)
    ax.legend(fontsize=28)
    ax.set_xlabel(r"step velocity (\textmu m/s)", fontsize=30)
    ax.set_ylabel("density", fontsize=30)
    ax.grid(False)   

plt.tight_layout()
if publish:
    pub.save_figure("best_sim_walking_lvel_distrib", notename)
    thesis.save_figure("best_sim_walking_lvel_distrib", notename)

# %%
df = abcimplement.tabulate_inference(new4dw["problem"], prime_accepted, "fanjin walking")
if publish:
    pub.pickle_df(df, "bayes_inference", notename)


# %%
df

# %%
from pili.support import make_get, get_array
target = join(pili.root, "../run/825bd8f/line_search/tdwell_search")
dc = rtw.DataCube(target=target)
ldlist = dc.load_local()
meanlvel = get_array(make_get("lvel.mean"), ldlist)

# %%
epsilon = _accepted["score"].max()

# %%
accparams = new4dw["params"].iloc[_accepted.index]
# plot lvel.mean vs tdwell
with mpl.rc_context({'font.size': 28, 'text.usetex':True}):
    fig, ax = plt.subplots(figsize=(5,5))
    mscat = {'edgecolor':'white', 'linewidth' : 1.0, 'alpha' : 0.8, 's': 100}
    m0 = {'c':'indianred', 'marker':'D', 'alpha':1.0, 'markersize':8}
    ax.scatter(accparams["dwell_time"], accparams["lvel.mean"], **mscat)

    # sns.scatterplot(data=accparams, x="dwell_time", y="lvel.mean", ax=ax)
    # ax.plot(m_par["dwell_time"], reference["lvel.mean"], **m0)
    ax.set_ylim((0,None))
    ax.set_xlim((0.1,3.0))
    ax.set_ylabel(r'$\langle u \rangle$~(\textmu m/s)', fontsize=32)
    ax.set_xlabel(r'$\tau_{\mathrm{dwell}}~(s)$', fontsize=32)

    ax.plot(dc.basis[0], meanlvel, **m0)
    accept_condition = r'$||\mathbf{s} - \mathbf{s}^\prime|| < 0.93$'
    ax.legend(["accepted", "line search"], fontsize=24)
    # ax.legend([accept_condition , "line search"])

pub.save_figure("abc_tdwell_vs_lvel", notename)

# %%
names = new4dw["problem"]["names"]
# _accepted[_objectives].corr()
# _accepted[names].corr()
_accepted.corr()

# %%
# params = new4dw["params"]
# sns.scatterplot(data=params, x="dwell_time", y="lvel.mean")


# %%
# -----------------------------------------------------------------------------
# copy from crawling abc.py
# the pdisp figure


# # %%
# # pull the best simulation
# import readtrack

# udir = join(new4dw["simdir"], new4dw["lookup"][0][_accepted.index[0]])
# twutils.sync_directory(udir)
# args = parameters.thisread(directory=udir)
# _trs = readtrack.trackset(ddir=join(udir, 'data/'))
# bestsim = [_fj.linearize(tr) for tr in _trs]

# # %%
# # load the xydistrib from the best simulation
# xydisp = np.load(join(udir, "xydisp.npy"))
# xydisp = xydisp[xydisp > 0]

# hstyle = {'stat':'density', 'alpha':0.3, 'fill':True, 'element':'step'}
# with plt.rc_context({'text.usetex': True, 'font.size':20}):
#     fig, ax = plt.subplots(figsize=(5.5, 5))
#     def plot_pdisp_distrib(ax, xydisp, hstyle):
#         sns.histplot(xydisp, ax=ax, **hstyle)
#         vstyle = dict(c='k', alpha=0.6,  linestyle='--')
#         ax.axvline(0.12, **vstyle)
#         ax.set_xlabel(r"per TFP displacement $(\mu m)$")
#         ax.set_xlim((0.,0.3))
#     plot_pdisp_distrib(ax, xydisp, hstyle)
# best_xyd = np.mean(xydisp)

# # %%

# simdir = new4dw["simdir"]

# def save_accepted_samples(simdir):
#     accept_uid = _accepted["uid"].to_numpy()
#     cachefile = join(simdir, "standard_accepted_uid.npy")
#     print("saving accepted uid list to ", cachefile)
#     np.savetxt(cachefile, accept_uid, fmt="%s")
# save_accepted_samples(simdir)


# # %%

# # pull the xydisp.npy data for the accepted samples
# def load_accepted_uid(simdir):
#     return np.loadtxt(join(simdir, "standard_accepted_uid.npy"), dtype=object)

# def load_accepted_xydisp(simdir):
#     uidlist = load_accepted_uid(simdir)
#     data = [np.load(join(simdir, uid, 'xydisp.npy')) for uid in uidlist]
#     data = list(map(lambda arr: arr[arr>0], data))
#     return data

# # load_accepted_uid(simdir)
# uidlist = load_accepted_uid(simdir)
# xydispdata = load_accepted_xydisp(simdir)


# # %%
# line_search = False

# if line_search:
#     target = join(pili.root, "../run/825bd8f/line_search")
#     dc = rtw.DataCube(target=target)
#     xydata = [np.load(join(directory, "xydisp.npy")) for directory in dc.dircube]
#     pdisp = [np.mean(xyd[xyd> 0]) for xyd in xydata]


# # %%
# means = [np.mean(xyd) for xyd in xydispdata]
# np.mean(means)

# # %%

# x = _accepted["anchor_angle_smoothing_fraction"].to_numpy()
# with mpl.rc_context({'text.usetex':True, 'font.size':20}):
#     fig, ax = plt.subplots(figsize=(4,4))
#     mscat = {'edgecolor':'white', 'linewidth' : 1.0, 'alpha' : 0.8, 's': 60}
#     ax.scatter(x, means, **mscat)
#     ax.set_ylim((0,None))
#     ax.set_xlim((0,np.pi/2))
#     ax.set_xticks([0,np.pi/4,np.pi/2])
#     ax.set_xticklabels(['0',r'$\pi/4$',r'$\pi/2$'])

#     # ax.scatter(m_par["anchor_angle_smoothing_fraction"], best_xyd)

#     m0 = {'c':'indianred', 'marker':'D', 'alpha':1.0, 'markersize':4}
#     if line_search:
#         base = np.array(dc.basis[0]) * np.pi/2
#         ax.plot(base, pdisp, **m0)

#     # ax.scatter(_accepted.iloc[0]["anchor_angle_smoothing_fraction"], best_xyd, marker='X', edgecolor='white', s=140)

#     ax.legend(["accepted", "line search", "best score"], loc='upper right', bbox_to_anchor=(1.0, 0.95), fontsize=11)

#     ax.set_ylabel(r'$\langle \Delta x \rangle$ $(\mu m)$')
#     ax.set_xlabel(r'$\alpha$')

#     inset = ax.inset_axes([0.50, 0.20, 0.6, 0.4])
#     def _plot(ax, xydisp, hstyle):
#         vstyle = dict(c='k', alpha=0.6,  linestyle='--')
#         orange = "#FF7F0E"
#         sns.histplot(xydisp, ax=ax, color=orange, **hstyle)
#         ax.axvline(0.12, **vstyle)
#         ax.set_xlabel(r"$ \Delta x$ $(\mu m)$", fontsize=12)
#         ax.set_ylabel(r"density", fontsize=12)
#         ax.set_xlim((0.,0.3))
#         # ax.annotate(r'$\delta_{\text{step}}', (0.4,0.5), xytext=(0.6, 0.5), textcoords='axes fraction')
#         ax.annotate(r'$\delta_{\mathrm{step}}$', (0.12, 10), xytext=(0.6, 0.5), textcoords='axes fraction', fontsize=16, arrowprops=dict(facecolor='black', width=1, shrink=0.05, headwidth=4))

#         # ax.annotate('{:.3f}'.format(ratio), (ratio,0), xytext=(0.7,0.5), textcoords='axes fraction', arrowprops=dict(facecolor='black', width=2, shrink=0.05)) 

#     _plot(inset, xydisp, hstyle)

# pub.save_figure('crawling_accepted_mean_pdisp', notename)
