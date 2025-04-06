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
# Implement metrics to compare two arbitrary distributions and apply 
# to velocity profiles.
# Check the implementations for similarity by eye
# At the end we can add the velocity distribution similarity metric
# to our analysis (e.g. sobolnote.py)

# %% 
import sys, os
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pili
import rtw
import _fj
import plotutils
import collections
import scipy.stats
import twanalyse
import pandas as pd
import parameters
import seaborn as sns


# %% 
# paths
notedir, notename = os.path.split(os.getcwd())
notedir, notename
root = pili.root
# candidate to compare against
print("loading experiment data")
all_idx, all_trs = _fj.slicehelper.load_linearized_trs("all")
flipped, scores = _fj.redefine_poles(all_trs)
reference_idx = _fj.load_subset_idx()
reftrs = {}
for key, subidx in reference_idx.items():
    reftrs[key] = [all_trs[idx] for idx in subidx]

print("finished")

# %% 
# simulation
angle1d_dir = join(root, "../run/new/angle_smoothed/range_pbrf")
simdata = collections.OrderedDict()
simdata[angle1d_dir] = rtw.DataCube(target=angle1d_dir)

# %% 
# config
histstyle = {'rwidth': 0.9}

# %% 
# plot individual track and combined distributions
fig,ax = plt.subplots(figsize=(6,4))
vellst = []
xlim = (0,1.0)
ax.set_title("whitelist tracks {}".format(len(reftrs["top"])))
for tr in reftrs["top"]:
    _vel = tr.get_step_speed()
    vellst.append(_vel)
    plotutils.ax_kdeplot(ax, _vel, xlims=xlim)
    ax.set_xlabel("step velocity")
    ax.set_ylabel("P")

fig,ax = plt.subplots(figsize=(6,4))
ax.set_title("whitelist tracks combined")
ref_vel = np.concatenate(vellst)
ax.hist(ref_vel, bins=20, range=xlim, **histstyle)
ax.set_xlabel("step velocity")
ax.set_ylabel("P")

plt.show()

# %% 
# plot velocity distributions for this 1d search
import readtrack
dc = list(simdata.values())[0]
print(str(dc))
trdata = dc.autocalculate(readtrack.trackset)
trdata = [[_fj.linearize(tr) for tr in trs] for trs in trdata]
vel = [np.concatenate([tr.get_step_speed() for tr in trs]) for trs in trdata]
nsteps = [np.sum([len(tr.step_idx) for tr in trs]) for trs in trdata]
print("nsteps", nsteps)
basis = dc.basis[0]

n = len(basis)
fig, axes = plt.subplots(n, figsize=(6,n*4))
# for i, value in list(enumerate(basis))[3:]:
for i, ax in enumerate(axes):
    plotutils.ax_kdeplot(ax, vel[i], xlims=xlim ,hist=True)

# %% 
# superimise one simulated trajectory with the reference data
# def mean(x): return scipy.stats.trim_mean(x, 0.025)
mean = np.mean

def plot_similarity(ax, sim_vel, ref_vel):
    v1 = sim_vel - mean(sim_vel)
    v2 = ref_vel - mean(ref_vel)
    show_hist = True
    plotutils.ax_kdeplot(ax, v1, xlims=(-0.5,0.5), hist=show_hist)
    plotutils.ax_kdeplot(ax, v2, xlims=(-0.5,0.5), hist=show_hist)
    ax.legend(["simulated", "reference"])
    ks_statistic, pvalue = scipy.stats.ks_2samp(v1, v2)
    chi = twanalyse.chisquare(v1, v2)
    ax.set_title("chi = {:.2f}, ks = {:.2f}".format(chi, ks_statistic))

# test
i = 2
fig, ax = plt.subplots(figsize=(6,4))
plot_similarity(ax, vel[i], ref_vel)

# %% 
# superimpose simulated data on reference data for the whole 1d range

n = len(basis)
fig, axes = plt.subplots(n, figsize=(6,n*4))
# for i, value in list(enumerate(basis))[3:]:
for i, ax in enumerate(axes):
    plot_similarity(ax, vel[i], ref_vel)
plt.tight_layout()

# %% [markdown]
# If we need a reference for what these similarity numbers actually
# mean we can check back on this notebook
# We should be ready to add these metrics to our summary statistics

# %% 
# switch over to searching sobol dataset for the closest examples
import sobol
import twutils
simdir = "/home/dan/usb_twitching/run/b2392cf/cluster/sobol_01"
lookup = sobol.read_lookup(simdir)
problem = sobol.read_problem(simdir)
twutils.print_dict(problem)
_ , lduid = sobol.collect([], targetdir=simdir, alldata=True)

# %% 
# load exp data
def _load_subset_speed():
    distrib = {}
    for name, ltrs in _fj.load_subsets().items():
        distrib[name] = np.concatenate([ltr.get_step_speed() for ltr in ltrs])
    return distrib
ref_vel = _load_subset_speed()

# %% 

subsets = reference_idx.keys()
# scores = ['fanjin.%s.chi' % subset for subset in reference_idx.keys()]
scores = ['fanjin.%s.ks_statistic' % subset for subset in reference_idx.keys()]
Yf = sobol.collect_obs(lookup, lduid, subsets, scores)
sortdf = sobol.sortscore(problem, lookup, Yf, scores)
sortdf["top"]

# TODO cache this data for use with other notebooks?

# %% 
# sync target data from cluster here in notebook
from sobol import sync_directory
best = sortdf["top"].iloc[0]

# %% 
mpl.rcParams["text.latex.preamble"] = r'\usepackage{booktabs}'
histstyle = {"stat":"density", "common_norm": False, "element":"step"}
def plot_superimposed(dfrow, subset, simdir, histstyle=histstyle):
    target = join(simdir, dfrow["dir"])
    if not os.path.exists(join(target, "data/")):
        output = sync_directory(target)
    ltrs = twanalyse.get_linearised_data(ddir=target)
    lvel = np.concatenate([ltr.get_step_speed() for ltr in ltrs])
    xlim = (0,1.2)
    data = {"sim": lvel, subset: ref_vel[subset]}
    fig, ax = plt.subplots()
    sns.histplot(data, binrange=xlim, ax=ax, **histstyle)
    ax.text(.5,.5, dfrow.to_latex().replace('\n', ' '),
        transform=ax.transAxes, fontsize=20)
    ax.set_xlabel("$\mu ms^{-1}$")
    return ax

def plot_subset_best(sortdf, simdir):
    for subset in sortdf.keys():
        i = 0
        best = sortdf[subset].iloc[i]
        plot_superimposed(best, subset, simdir, histstyle=histstyle)
# %% 
plot_subset_best(sortdf, simdir)

# %% 
use_chi = False
if use_chi:
    chi_scores = ['fanjin.%s.chi' % subset for subset in reference_idx.keys()]
    chi_Yf = sobol.collect_obs(lookup, lduid, subsets, chi_scores)
    chi_sortdf = sortscore(problem, lookup, chi_Yf, chi_scores)
    chi_sortdf["candidate"]
# %% 
if use_chi:
    plot_subset_best(chi_sortdf, simdir)

# %% 
# It's clear our chi similarity totally fails for "top" and "candidate" 
# but we know better matches exist because the ks_statistic works much better
import scipy.stats
check_chi = False
if check_chi:
    _best = chi_sortdf["top"].iloc[0]
    res = 100
    ltrs = twanalyse.get_linearised_data(ddir=join(simdir, _best["dir"]))
    lvel = np.concatenate([ltr.get_step_speed() for ltr in ltrs])
    ref = ref_vel["top"]
# %%
if check_chi:
    v1 = lvel - np.mean(lvel)
    v2 = ref - np.mean(ref)
    print("mean", np.mean(lvel), np.mean(ref))
    _q = 0.050 # vary this
    xn1, xm1 =  np.quantile(v1, _q), np.quantile(v1, 1.0 - _q)
    xn2, xm2 =  np.quantile(v2, _q), np.quantile(v2, 1.0 - _q)
    xn, xm = min(xn1, xn2), max(xm1, xm2)
    print(xn1, xm1)
    print(xn2, xm2)
    print("xlims", xn, xm)
    mspace = np.linspace(xn, xm, res)
    # method = "scott"
    def method(self):
        div_f = 4.0 # vary this
        return np.power(self.neff, -1./(self.d+4)) / div_f
    kde1 = scipy.stats.gaussian_kde(v1, bw_method=method)
    kde2 = scipy.stats.gaussian_kde(v2, bw_method=method)
    pde1 = kde1.evaluate(mspace)
    pde2 = kde2.evaluate(mspace)
    plt.plot(mspace, pde1, label="")
    plt.plot(mspace, pde2)
    chisquared = np.sum((pde2 - pde1)**2/(pde1 + pde2))
    print("chi", np.sqrt(chisquared))

    fig, ax = plt.subplots()
    sns.histplot({"sim":v1, "top":v2}, binrange=(xn, xm), **histstyle)

# %% [markdown]
# the chi metric is failing because the bandwidth is too large
# reducing by a factor 4 works well for this example but it may make the other 
# examples worse (?)
# until we can figure out a more robust method, put trust in ks_statistic instead

# %% 
simdir = "/home/dan/usb_twitching/run/5bfc8b9/cluster/sobol_walking"
lookup = sobol.read_lookup(simdir)
problem = sobol.read_problem(simdir)
print(problem)
_ , lduid = sobol.collect([], targetdir=simdir, alldata=True)
# %% 
import copy
_lookup = copy.deepcopy(lookup)
_lduid = copy.deepcopy(lduid)
for i, uid in reversed(list(enumerate(lookup[0]))):
    ld = lduid[uid]
    if ld.get("failed", False):
        print (uid, "failed", ld["failed_condition"])
        del _lduid[uid]
        del _lookup[1][uid]
        del _lookup[0][i]

# %% 
print(len(_lookup[0]), len(_lookup[1]), len(_lduid))
# %% 

scores = ['fanjin.%s.ks_statistic' % subset for subset in reference_idx.keys()]
Yf = sobol.collect_obs(_lookup, _lduid, subsets, scores)
sortdf = sobol.sortscore(problem, _lookup, Yf, scores)
sortdf["walking"]

# %% 
best = sortdf["walking"].iloc[0]
_style = copy.deepcopy(histstyle)
# _style["kde"] = True
ax = plot_superimposed(best, "walking", simdir, histstyle=_style)
plt.tight_layout()
plt.savefig("/home/dan/usb_twitching/notes/sensitivity/best_walking.png")
print("best simulation at ", join(simdir, best["dir"]))

# %% [markdown]
# best simulation
# `/home/dan/usb_twitching/run/5bfc8b9/cluster/sobol_walking/_u_cs5aMoaJ`

# %% 
# break down the fanjin walking tracks  
subsets = _fj.load_subsets()
subset_idx = _fj.load_subset_idx()
# %% 
# %% 
walking = subsets["walking"]
each = [w.get_step_speed() for w in walking]
xlim = (0,3.0)
fig, axes = plt.subplots(35, 5, figsize=(5*5,35*5))
for i, ax in enumerate(axes.ravel()):
    # ax = sns.kdeplot(each[0], bw_adjust=0.5)
    sns.histplot(each[i], binrange=xlim, ax=ax)
    ax.set_xlim(xlim)
    ax.set_title(str(i))
plt.tight_layout()
# plt.savefig "all_walking_lvel.svg"

# %%
# cherry pick the  trajectories which show the 3 peak behaviour seen in the combined distribution
cherry_idx = [35, 125, 129, 165]
cherry= [walking[i] for i in cherry_idx]
for c in cherry:
    fig, ax = plt.subplots(figsize=(5,5))
    sns.histplot(c.get_step_speed(),  binrange=[0,2])
# %%
import fjanalysis
ldcherry = [fjanalysis.lsummary([c]) for c in cherry]
_interest = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'kmsd.mean']
for var in _interest:
    getter = twutils.make_get(var)
    v = [getter(ld) for ld in ldcherry]
    print(var, v)
# compare to the whole reference set
print('---------')
ldref = fjanalysis.lsummary(walking)
{var : twutils.make_get(var)(ldref) for  var in _interest}
# notice that track 125 is close to all the averages for the whole set (it is also the longest)
# take this trajectory has our representative  
# %%
wrep = walking[125]
track_id = subset_idx['walking'][125]
print('representative walking track', track_id)

# %%
