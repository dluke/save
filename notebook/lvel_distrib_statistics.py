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
# look in the experimental data and then the simulated data for peaks in the velocity distributions
# and develop a robust fitting procedure for these peaks

# %%
import os
import json
import numpy as np
import scipy.stats
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd

import command
import readtrack
import parameters
import _fj
import fjanalysis
import stats

from pili import support
import pili

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import sobol
import abcimplement
import mdl
import pwlpartition
import pwlstats

import pili.publication as pub
print("writing figures to", pub.writedir)


from skimage.restoration import denoise_wavelet, estimate_sigma

from support import make_get, get_array

# %%
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
notename = 'sim_pwl_vretabc'
mplstyle = {"font.size": 20}

# %% 
subsets = _fj.load_subsets()
subset_idx = _fj.load_subset_idx()

# %% 
toptracks = subsets['top']
candidate = subsets['candidate'][0]

# lvel = toptracks[0].get_step_speed()
lvel = candidate.get_step_speed()

with mpl.rc_context({'font.size': 16}):
    fig, ax= plt.subplots(figsize=(6,4))
    shstyle = dict(stat='density', element="step", fill=False)
    xlim = (0,2.0)
    sns.histplot(lvel, ax=ax, binrange=xlim, **shstyle)

save_lvel = lvel

# %% 
# * we can't see two peaks, what if I use my own version of the wavelet transform ...
# first load the orginal data
path =  join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
sigma, r = pwlstats.load_candidate_sigma_r(path)
original_subsets = _fj.load_subsets(original=True)

# %% 

denoise_candidate = pwlstats.denoise_track(original_subsets['candidate'][0], sigma)

# without linearisation
# lvel = denoise_candidate.get_step_speed()
# with linearisation
dstep = 0.06
lin_candidate = _fj.linearize(denoise_candidate, step_d=dstep)
lvel = lin_candidate.get_step_speed()

fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(lvel, ax=ax, binrange=xlim, **shstyle)


# %%
print(save_lvel.size, lvel.size)
with mpl.rc_context({'font.size': 14}):
    fig, ax = plt.subplots(figsize=(6,4))
    support.compare_distrib(ax, save_lvel, lvel)
    ax.legend(['FJ wavelet lvel', 'sklearn wavelet lvel,'+f' $\delta = {dstep}$'])

# %%
# so lets try extracting an lvel distribution from the PWL model

path =  join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
solver = pwlpartition.Solver.load_state(join(path, 'solver'))
model, data = solver.get_model(), solver.get_data()

curve_coord = solver.partition.get_curve_coord()

# * this denoising has little effect here so don't complicate
_sigma = estimate_sigma(curve_coord)
wave_config = {"wavelet":'db1', 'method':'BayesShrink', "mode":'soft', "rescale_sigma":False}
wave_curve_coord = denoise_wavelet(curve_coord, sigma=_sigma, **wave_config)


# %%
# count mapped data using dstep, this is similar to linearisation
dstep = 0.06
def mapped_velocity(curve_coord, dstep):
    travel = 0
    idx = 0
    count_mapped = []
    step_distance = []
    while True:
        if travel > curve_coord[-1]:
            break
        # counting
        next_travel = travel + dstep
        next_idx = np.searchsorted(curve_coord, next_travel)
        count = next_idx - idx
        if count > 0:
            count_mapped.append(count)
            step_distance.append(dstep)
        else:
            next_travel = curve_coord[next_idx+1]
            next_idx = next_idx+1
            count_mapped.append(1)
            step_distance.append(next_travel - travel)
        idx = next_idx
        travel = next_travel

    step_distance= np.array(step_distance)
    count_mapped = np.array(count_mapped)
    return count_mapped, step_distance

# count_mapped, step_distance = mapped_velocity(curve_coord, dstep)
count_mapped, step_distance = mapped_velocity(curve_coord, dstep)

_DT = 0.1
mapped_lvel = step_distance/(_DT * count_mapped)
print('check', np.sum(count_mapped==0), np.sum(step_distance > dstep))
print('mapped_lvel.size', mapped_lvel.size)

with mpl.rc_context({'font.size': 12}):
    fig, ax = plt.subplots(figsize=(6,4))
    _xlim = xlim
    support.compare_distrib(ax, lvel, mapped_lvel, style={'stat':'count', 'binrange':_xlim})
    ax.legend(['sklearn wavelet lvel', 'PWL mapped velocity,'+f' $\delta = {dstep}$'])

# %%
# check the counting distribution

with mpl.rc_context({'font.size': 12}):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(count_mapped)
    ax.set_xlabel("count mapped data")


# %%
# * ------------------------------------------------------------
# do the same for a slower track


# %%
dstep = 0.06

top_idx =  0

denoise_track = pwlstats.denoise_track(original_subsets['top'][top_idx], sigma)
lin = _fj.linearize(denoise_track, step_d=dstep)
local = fjanalysis.lsummary([lin])
N = lin.get_nsteps()
print('track ', subset_idx['top'][top_idx], 'vel', local['lvel']['mean'], 'N steps', N)

track_idx = subset_idx['top'][top_idx]

this_lvel = lin.get_step_speed()
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(this_lvel, ax=ax, binrange=xlim, **shstyle)

# %%
# load the PWL attempt
# ! this solve still has problems at the breakpoints
solver = pwlpartition.Solver.load_state(join(pwlstats.root, f"run/cluster/no_heuristic/top/_top_{track_idx:04d}/", "solver"))
this_curve_coord = solver.partition.get_curve_coord()

count_mapped, step_distance = mapped_velocity(this_curve_coord, dstep)
this_mapped_lvel = step_distance/(_DT * count_mapped)

with mpl.rc_context({'font.size': 12}):
    fig, ax = plt.subplots(figsize=(6,4))
    _xlim = xlim
    support.compare_distrib(ax, this_lvel, this_mapped_lvel, style={'stat':'count', 'binrange':_xlim})
    # ax.legend(['sklearn wavelet lvel', 'PWL mapped velocity,'+f' $\delta = {dstep}$'])
    inset = ax.inset_axes([0.55, 0.45, 0.4, 0.4])
    inset.set_xlabel("count mapped data")
    sns.histplot(count_mapped, ax=inset)
    ax.set_xlabel(r"velocity $(\mu m /s)$")


# %%
this_mapped_lvel[count_mapped==1]


# %%
# * ------------------------------------------------------------
# do the same for simulated data

