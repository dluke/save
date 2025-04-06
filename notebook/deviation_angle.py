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
# The objective of this notebook is just to setup the necessary codes to run simulations 
# with variable parameters: pilivar, anchor_smoothing, dwell_time and k_spawn.
# 
# the primary metric we will use is the width of the deviation angle distribution
# (see fanjin.py)

# analysis of deviation angle in fanjin datasets

# %%
import sys,os
import numpy as np
join = os.path.join
import twanalyse
import readtrack
import command
import _fj
import rtw
import matplotlib.pyplot as plt
import scipy.stats
import rtw
import fjanalysis
from tabulate import tabulate
# %%
rundir = "/home/dan/usb_twitching/run/"
simdir = join(rundir, "b2392cf/pilivar")

def lintrackset():
	return [_fj.linearize(tr) for tr in  readtrack.trackset()]
with command.chdir(simdir):
    dc = rtw.DataCube()
    trcube = dc.autocalculate(lintrackset, force=False)
    ldata = dc.load_local()
print('loaded data at ', simdir)

# %%
print(dc.basis)
# %%
# get one track
trs = trcube[0]

# compute deviation angle distribution and then the width of that
pdata = twanalyse.allpolar(trs)
fig, axes = plt.subplots(3, 2, subplot_kw=dict(polar=True), 
    figsize=(8,12))
twanalyse.plotpolar(axes, pdata)

# %%
# %%
# plot deviation distributions on vertical subplots
import seaborn as sns

n = trcube.size
fig, axes = plt.subplots(n, 1, figsize=(5, n*2))
for i ,ax in enumerate(axes):
    pdata = twanalyse.allpolar(trcube[i])
    describe = scipy.stats.describe(pdata.deviation)
    xlim = (-np.pi,np.pi)
    ax = sns.histplot(pdata.deviation, ax=ax, binrange=xlim)
    title = "pilivar = {:.2f}, 'var = {:.3f}".format(
        dc.basis[0][i], describe.variance)
    ax.set_title(title)
    ax.set_xlabel("deviaton angle")
plt.tight_layout()
plt.show()

# %% [markdown]
# we see that deviation.var in simulation can be as high as 1.5
# just like our "median" crawling datasets

# and we notice a clear transition between deviation angles 
# being anticorrelated and correlated with the polar axis

# %% 
use_uniform = False
if use_uniform:
    simdir = join(rundir, "new/pilidistrib/uniform")
    with command.chdir(simdir):
        dc = rtw.DataCube()
        trcube = dc.autocalculate(lintrackset, force=False)


# %% 
if use_uniform:
    n = trcube.size
    fig, axes = plt.subplots(n, 1, figsize=(10, n*5))
    for i ,ax in enumerate(axes):
        pdata = twanalyse.allpolar(trcube[i])
        ax.set_title(dc.basis[0][i])
        ax.hist(pdata.deviation)
        ax.set_xlim(-np.pi/2,np.pi/2)
    plt.show()

# %% [markdown]
# so actually, no pili on the body cylinder are not the cause
# it is pili with anchors > pi/4 away from the body axis that seem 
# to cause deviation angles anticorrelated with the body axis
# many such pili will be short and quickly become taut, 
# is the bacterium actually progressing forwards for these parameters
# sets or is it just oscillating?
# we would want to check the 
# * total displacement of the body 
# * no. bound pili

# %%
# candidate track is
candidate = 2924
candidate_track = _fj.lintrackload([2924])[0]
pdata = twanalyse.allpolar([candidate_track])
vel = twanalyse._inst_vel(candidate_track)
candidate_lvel = scipy.stats.describe(vel)
lvel_sem = scipy.stats.sem(vel)
sd = twanalyse._qaparams([candidate_track])


# %%
velocities, deviations = twanalyse.get_candidate_distributions()
# np.var(deviations)

# %%
# styles
style = {'marker':'o'}
hstyle = {'linestyle':'--', 'color':'k', 'alpha':0.6}

# %%
# average velocity

def _velocity(ax):
    v = [ld['lvel']['mean'] for ld in ldata]
    v_err = np.array([ld['lvel']['std_error'] for ld in ldata])
    ax.errorbar(dc.basis[0],v, 1.96*v_err, **style)
    lvel_mean = candidate_lvel.mean
    ax.axhline(lvel_mean, **hstyle)
    error = 1.96 * lvel_sem
    ax.axhspan(lvel_mean-error,lvel_mean+error, color='k', alpha=0.2)
    ax.set_ylabel('mean velocity')
    ax.set_xlabel('pili distribution width')
    ax.set_ylim(0.0,0.3)

# %%
# deviation
def _deviation(ax):
    candidate_var = np.var(pdata.deviation)
    v = [ld['deviation']['var'] for ld in ldata]
    ax.plot(dc.basis[0],v, **style)
    ax.set_ylabel('mean velocity')
    ax.set_xlabel('pili distribution width')
    ax.axhline(candidate_var, **hstyle)
    ax.set_ylabel(r'var(deviation angle)')
    ax.set_xlabel('pili distribution width')
    ax.set_ylim(0,2.0)


# %%
# persistance
def _macro(ax, stat='qhat', ylabel='persistence'):
    candidate_q = sd[stat]['estimate']
    candidate_q_err = sd[stat]['err']
    error = 1.96 * candidate_q_err

    v = [ld[stat]['estimate'] for ld in ldata]
    v_err = np.array([ld[stat]['err'] for ld in ldata])

    ax.errorbar(dc.basis[0],v, 1.96*v_err, **style)

    ax.axhline(candidate_q, **hstyle)
    ax.axhspan(candidate_q-error,candidate_q+error, color='k', alpha=0.2)
    ax.set_ylabel('mean velocity')
    ax.set_xlabel('pili distribution width')
    ax.set_ylim(0.0,1.0)

    ax.set_ylabel(ylabel)
    ax.set_xlabel('pili distribution width')


# %%
# plot togethers
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = '26'
fig, axes = plt.subplots(4,1,sharex=False, figsize = (5,2*5))
ax1, ax2, ax3, ax4 = axes
_velocity(ax1)
_deviation(ax2)
_macro(ax3)
_macro(ax4, stat='ahat', ylabel='activity')
plt.tight_layout()
plt.show()

# %% [markdown]

# %% 
# ok but lets now examine the deviation angle for each dataset
# load fanjin data
all_idx, all_ltrs = _fj.slicehelper.load_linearized_trs("all")

# %% 
# COPIED sobolnote.py
objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
getter = [rtw.make_get(name) for name in objectives]

reference_idx = _fj.load_subset_idx()
# refdata: {subset: {objective: weighted mean value}}
refdata = {}
reftrs = {}
localdata = {}
for key, subidx in reference_idx.items():
    reftrs[key] = [ltrs[idx] for idx in subidx]
    localdata[key] = fjanalysis.lsummary(reftrs[key])

for subset, data in reftrs.items():
    nsteps = [len(ltr.step_idx)-1 for ltr in data]
    ldata = localdata[subset]
    refdata[subset] = {}
    for i, objective in enumerate(objectives):
        ref_value = getter[i](ldata)
        refdata[subset][objective] = ref_value
    
rows = [[subset]+[refdata[subset][name] for name in objectives] 
    for subset in refdata.keys()]
table = tabulate(rows, headers=['-'] + objectives)
print(table)

# %% 
# now check the persistence values of each subset
distrib_names = ['top', 'half', 'median', 'walking']
q_distrib = {}
a_distrib = {}
for name, ltrs in reftrs.items():
    qap = [twanalyse._qaparams([ltr]) for ltr in ltrs]
    q = [ld['qhat']['estimate'] for ld in qap]
    a = [ld['ahat']['estimate'] for ld in qap]
    q_distrib[name] = q
    a_distrib[name] = a
    
import seaborn as sns

n = len(distrib_names)
fig, axes = plt.subplots(n, figsize=(6,n*3))

xlim = (-1.0,1.0)
for i, name in enumerate(distrib_names):
    ax = axes[i]
    sns.histplot(q_distrib[name], ax=ax, binrange=xlim)
    ax.set_ylabel('persistence')
    ax.set_title(name)
plt.tight_layout()
# %% 
# we see the peristence of all the crawling tracks is positive or close to zero
# and only walking tracks have negative persistence
