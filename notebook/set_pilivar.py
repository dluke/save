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
# plot deviation distributions on vertical subplots

n = trcube.size
fig, axes = plt.subplots(n, 1, figsize=(10, n*5))
for i ,ax in enumerate(axes):
    pdata = twanalyse.allpolar(trcube[i])
    describe = scipy.stats.describe(pdata.deviation)
    print('var = ', describe.variance)
    ax.hist(pdata.deviation)
    ax.set_xlim(-np.pi/2,np.pi/2)
plt.show()

# %% [markdown]
# and we notice a clear transition between deviation angles 
# being anticorrelated and correlated with the polar axis
# Intuition is that this is due to pili on the body section
# we can test this hypothesis using uniform distribution

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
# -- we would want to check the 
#   - total displacement of the body 
#   - no. bound pili

# %%
# candidate track is
candidate = 2924
candidate_track = _fj.lintrackload([2924])[0]
pdata = twanalyse.allpolar([candidate_track])
vel = twanalyse._inst_vel(candidate_track)
candidate_lvel = scipy.stats.describe(vel)
lvel_sem = scipy.stats.sem(vel)

# plt.hist(pdata.deviation)
sd = twanalyse._qaparams([candidate_track])

step_vel =  np.linalg.norm(candidate_track.get_step_velocity(), axis=1)
plt.hist(step_vel)

# %%
velocities, deviations = twanalyse.get_candidate_distributions()
np.var(deviations)

# %%
for ltrs in trcube:
    linvel = np.concatenate([np.linalg.norm(ltr.get_step_velocity(),axis=1) for ltr in ltrs])
    plt.hist(linvel)
    print(np.mean(linvel))
    # TODO
    # mean of step velocity and the normal mean velocity cannot be this different
    # check for bugs

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
ax = plt.gca()
_velocity(ax)

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
ax = plt.gca()
_deviation(ax)


# %%
# persistance
ax = plt.gca()
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
_macro(ax)

# %%
# activity
ax = plt.gca()
_macro(ax, stat='ahat', ylabel='activity')

# %%
# plot togethers
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = '26'
fig, axes = plt.subplots(4,1,sharex=False, figsize = (5,4*5))
ax1, ax2, ax3, ax4 = axes
_velocity(ax1)
_deviation(ax2)
_macro(ax3)
_macro(ax4, stat='ahat', ylabel='activity')
plt.tight_layout()
plt.show()
