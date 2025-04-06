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
# analyse the summary statistics of the crawling trajectories to identify a "typical" trajectory
# with a long sampling time (for good statistics)

# the idea is that it is unnatural to average over bacteria, the way we have been doing
# once we identify a typical trajectory we can copy crawlingabc.py / simabc.py and 
# infer parameters

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

import twutils
from pili import support
from pili.support import make_get, get_array
import pili
import _fj
import fjanalysis
import abcimplement

import pili.publication as pub

# %% 

use_candidates_whitelist = False

if use_candidates_whitelist:
    # !use default crawling list?
    topidx = _fj.load_subset_idx()["top"]
    toptrs = _fj.load_subsets()["top"]
else:
    # !use the full data set 
    fjlocal = fjanalysis.load_summary()
    idx, crawling = _fj.slicehelper.load_linearized_trs('default_crawling_list')

    crdatalist = [fjlocal[i] for i in idx]
    crproc = pd.DataFrame({
        'duration' : [tr.get_duration() for tr in crawling],
        'nsteps': [tr.get_nsteps() for tr in crawling],
        'lvel.mean' : get_array(make_get('lvel.mean'), crdatalist)},
        index = idx
    )
    condition1 = crproc[crproc["nsteps"] > 100]
    condition2 = condition1[condition1["duration"] > 100]
    condition3 = condition2[condition2["lvel.mean"] > 0.05]
    topidx = condition3.index.to_numpy()
    toptrs = _fj.lintrackload(topidx)

# %%
durations = np.array([tr.get_duration() for tr in toptrs])
nsteps = np.array([tr.get_nsteps() for tr in toptrs])
print('N = ', len(toptrs))
print('min/max duration', durations.min(), durations.max())
print('min/max nsteps', nsteps.min(), nsteps.max())

# %%
# one by one
datalist =  [{**{'uid': idx}, **fjanalysis.lsummary([tr])} for idx, tr in zip(topidx,toptrs)]

# %%
observable = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"]
data = {obs : get_array(make_get(obs), datalist) for obs in  observable}
_refdf = pd.DataFrame(data, index=topidx)

# %%
# plot distributions

for i, obs in enumerate(observable):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(_refdf[obs])
    ax.set_xlabel(obs)

# %%
t_refdf = _refdf[_refdf["deviation.var"] < 1.0]
t_refdf = t_refdf[t_refdf["lvel.mean"] < 0.2]
t_refdf = t_refdf[t_refdf["ahat.estimate"] < 0.3]
refdf = t_refdf
# obsthreshold1 = refdf[refdf"deviation.var"] < np.pi]
# k means clustering with k = 2 to remove outliers/walking?

for i, obs in enumerate(observable):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(refdf[obs])
    ax.set_xlabel(obs)

# %%
len(refdf), len(_refdf)

# %%

# df.corr()

rescale_refdf = refdf/refdf.std()
rescale_refdf

# simply find the center of mass of the data
pts = rescale_refdf[observable].to_numpy()
center = np.mean(pts, axis=0)

# euclidean norm doesn't make sense unless we rescale the axes
rowid = np.argmin(np.linalg.norm(center-pts, axis=1))
print("special", rowid, refdf.index[rowid], "\n", refdf.iloc[rowid])
# !typical trackidx 2127
# !use this trackidx as reference data
special_rowid = rowid

# %%
# OK, just run ABC on every sample

# %%
new4d = {}
new4d["simdir"] = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d"
new4d["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'fanjin.top.ks_statistic', 'kmsd.mean', 'nbound.mean', 'ntaut.mean']
new4d = abcimplement.load_problem_simulation(new4d)

N = 50
new4d["params"] = new4d["data"].paramsdf(new4d["objectives"])
abcimplement.transform_anchor_parameter_data(new4d)


# %%
dflist = []
acclist = []

_objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
for index, reference in refdf.iterrows():
    statdf, statref = abcimplement.regularise_stats(new4d["params"], reference, new4d["objectives"])
    _regdf = statdf[["uid"] + new4d["problem"]["names"] + _objectives]
    _accepted = abcimplement.rejection_abc(_regdf, _objectives, statref, N)
    df = abcimplement.tabulate_inference(new4d["problem"], _accepted, _objectives)
    dflist.append(df)
    acclist.append(_accepted)

# %%

names = new4d["problem"]["names"]
mle_estimate = get_array(make_get("MLE"), dflist)
for name, col in zip(names, mle_estimate.T):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(col)
    ax.set_xlabel(name)

# %%

est_lower = get_array(lambda df: df["confidence(0.05)"], dflist)
est_upper = get_array(lambda df: df["confidence(0.95)"], dflist)

lower_size = mle_estimate - est_lower
upper_size = est_upper - mle_estimate

yerr = np.stack([lower_size,upper_size], axis=-1)

# %%
names = new4d["problem"]["names"]

# turn off this plot because the indexed one is better
if False: 
    lvel = refdf["lvel.mean"]

    for i, name in enumerate(names):
        _yerr = yerr[:,i,:].T

        fig, ax = plt.subplots(figsize=(6,4))
        ax.errorbar(lvel, mle_estimate.T[i], yerr=_yerr, linestyle='none', marker='D')

        ax.set_xlabel("lvel.mean")
        ax.set_ylabel("estimate "+ name)

# %%
# the same but using index as x-axis

for i, name in enumerate(names):
    _yerr = yerr[:,i,:].T
    N = len(mle_estimate.T[0])

    fig, ax = plt.subplots(figsize=(6,4))
    ax.errorbar(range(N), mle_estimate.T[i], yerr=_yerr, linestyle='none', marker='D')

    ax.errorbar(special_rowid, mle_estimate.T[i][special_rowid], yerr=yerr[special_rowid,i,:][:,np.newaxis], linestyle='none', marker='D', color='r')
    

    ax.set_xlabel("index")
    ax.set_ylabel("estimate "+ name)


# %%

# pull the trajectories with high \kappa estimate ~ what is special about them?

hkidx = np.argwhere(mle_estimate[:,1] > 8.0).ravel()
nhkidx = np.argwhere(~(mle_estimate[:,1] > 8.0)).ravel()
high_kappa_trackidx = topidx[hkidx]

# display their inference tables
# from IPython.display import display, HTML
# for idx in hkidx:
#     display(HTML(dflist[idx].to_html()))


highkappa = refdf.iloc[hkidx]
lowkappa = refdf.iloc[nhkidx]
for i, name in enumerate(_objectives):
    fig, ax= plt.subplots(figsize=(6,4))
    support.compare_distrib(ax, highkappa[name], lowkappa[name] )
    ax.legend(["high", "low"])

# %%
# * high persistance? so what do these trajectories look like?
notedir = "typical_crawling/"


defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_data(ax, x, y):
    ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.1, 'color':defcolors[2]}
    ax.plot(x, y, **ptlkw)
    ax.set_aspect("equal")

def plot_linear(ax, x, y):
    model_style = {"linestyle": '-', 'lw':1, 'alpha':0.8, 'color':defcolors[1]}
    ax.plot(x, y, marker='o', markersize='1', **model_style)
    ax.set_aspect("equal")
    
    

# plot these trajectories to a local directory
boxsizefactor = 2
for idx in hkidx:
    track = toptrs[idx]
    boxsize = track.get_bbox_size()
    figsize = boxsizefactor * boxsize
    print('using figsize', figsize)
    fig, ax = plt.subplots(figsize=figsize)

    # display(HTML(dflist[idx].to_html()))

    print(refdf.iloc[idx])
    trackidx = topidx[idx]
    _original = _fj.trackload_original([trackidx])[0]
    plot_data(ax, _original['x'], _original['y'])
    
    x, y = track['x'][track.step_idx], track['y'][track.step_idx]
    plot_linear(ax, x, y)

    target = join(notedir, f"high_kappa_track_{trackidx:04d}.svg")
    # print("saving to ", target)
    # fig.savefig(target)
    break

# %%
# okay and lets look at the best simulation

bestuid = acclist[hkidx[0]].iloc[0]['uid']
params = new4d["params"]
bestparams = params[params['uid'] ==  bestuid]
# syncing
sim_at = join(new4d["simdir"], bestuid)
twutils.sync_directory(sim_at)
bestparams

# %%
# 
track = toptrs[hkidx[0]]

import readtrack
simtrack = readtrack.trackset(ddir=join(sim_at, "data/"))[0]
simtrack.cut_time(0, track.get_duration())
fig, ax = plt.subplots(figsize=boxsizefactor * simtrack.get_bbox_size())
plot_data(ax, simtrack['x'], simtrack['y'])

# %%
# plot kspawn vs qhat.estimate
fix_dwell = params[params["dwell_time"].between(0.8,1.2)]
fix_alpha = fix_dwell[fix_dwell["anchor_angle_smoothing_fraction"].between(0.3,0.5)]

# sns.scatterplot(data=fix_alpha, x="k_spawn", y="qhat.estimate")
# sns.scatterplot(data=fix_alpha, x="k_spawn", y="kmsd.mean")
# scipy.stats.pearsonr(fix_alpha["k_spawn"], fix_alpha["kmsd.mean"])
# sns.scatterplot(data=fix_alpha, x="ntaut.mean", y="kmsd.mean")


# %%
fix_dwell = params[params["dwell_time"].between(0.8,1.2)]
fix_alpha = fix_dwell[fix_dwell["anchor_angle_smoothing_fraction"].between(0.3,0.5)]
sns.scatterplot(data=fix_alpha, x="k_spawn", y="kmsd.mean")

# %%
fix_alpha[names + ["qhat.estimate"]].corr()

# %%
#----------------------------------------------------------------
# * the chosen "typical" trajectory inference
dflist[special_rowid]



# %%
