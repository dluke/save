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
# use sklearn wavelet methods to 
# 1. analyse the spatial error in our candidate datasetse
# 2. replace fanjin oddly smooth wavelet transform with our own ( where we know the thresholding used)


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
import _fj
import fjanalysis
import stats
import twanalyse

from pili import support
import pili

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import mdl
import pwlstats
import pwlpartition

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

from support import make_get, get_array

from skimage.restoration import estimate_sigma, denoise_wavelet

# %%
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
notename = 'new_wavelet'
mplstyle = {"font.size": 20}

# %%
# load fanjin data subsets
subsets = _fj.load_subsets(original=True)
subset_idx = _fj.load_subset_idx()

fjlocal = fjanalysis.summary()

# %%
dctlist = []

for name, tracklist in subsets.items():
    lambda_sigma =  lambda track: pwlpartition.estimate_error(track['x'], track['y'])
    est_sigma = np.array(list(map(lambda_sigma, tracklist)))
    mean_est_sigma = np.mean(est_sigma)
    durations = [track.get_duration() for track in tracklist]
    N =  len(tracklist)
    print('name, ', name, 'N', N, 'sigma', mean_est_sigma)
    # print(est_sigma.min(), est_sigma.max())
    sd = {
        'name' : name,
        'N' : len(tracklist), 'sigma' :  mean_est_sigma,
        'est_sigma' :  est_sigma,
        'durations' :  durations
    }
    dctlist.append(sd)

# %%
ax = sns.histplot(dctlist[1]['durations'])

# %%
# plot estimated spatial error
with mpl.rc_context({'font.size': 12}):
    fig, ax = plt.subplots(figsize=(6,4))

    _cdctlist = dctlist[1:]

    l = ax.axvline(dctlist[0]['sigma'], c='k', linestyle='--', alpha=0.4, label='candidate')
    for sd in _cdctlist:
        sns.histplot(sd['est_sigma'], stat='count', element='step', fill=False, ax=ax, label=sd['name'])

    ax.legend()
    ax.set_xlabel(r'estimate $\sigma$')
pub.save_figure('estimate_sigma_fanjin_datasets', notename)

# %%
# plot estimated spatial error by weighted by duration

with mpl.rc_context({'font.size': 12}):
    fig, ax = plt.subplots(figsize=(6,4))

    _cdctlist = dctlist[1:]

    l = ax.axvline(dctlist[0]['sigma'], c='k', linestyle='--', alpha=0.4, label='candidate')
    for sd in _cdctlist:
        sns.histplot(x=sd['est_sigma'], bins=10, stat='count', element='step', fill=False, ax=ax, label=sd['name'],  weights=sd['durations'])

    ax.legend()
    ax.set_xlabel(r'estimate $\sigma$')

# pub.save_figure('estimate_sigma_fanjin_datasets', notename)


# %%
# pull the min/max sigma estimate for top data
toplist = dctlist[1]
minidx, maxidx = toplist['est_sigma'].argmin(), toplist['est_sigma'].argmax()
min_track, max_track = subsets['top'][minidx], subsets['top'][maxidx]
min_track.get_duration(), max_track.get_duration()
short_min_track = min_track.cut_time(0,20)
def new_local_plot_tracks(trs):
    fig, ax = plt.subplots(figsize=(12,5))
    for tr in trs:
        ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.2}
        ax.plot(tr['x'], tr['y'], **ptlkw)
        ax.set_aspect("equal")
    return ax 
new_local_plot_tracks([short_min_track])
new_local_plot_tracks([max_track.cut_time(0,20)])

# %%
# * set up the sklearn wavelet transform to be used as smooth fanjin data
use_sigma = dctlist[0]['sigma']
print("using sigma", use_sigma)
# ! we use the sigma from candidate track because we verified this one by eye
# todo. After solving, examine the residual distribution and check sigma

scikit_config = {"wavelet":'db1', 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}
def denoise_track(track, sigma, config=scikit_config):
    track =  track.copy()

    x = denoise_wavelet(track['x'], sigma=sigma, **config)
    y = denoise_wavelet(track['y'], sigma=sigma, **config)
    track['x'] = x
    track['y'] = y
    lmodel = pwlpartition.contract(np.stack([x,y]))
    step_idx = lmodel.get_time()
    step_idx[-1] -= 1 
    _fj.lin(track, step_idx)
    track.step_idx = step_idx

    x = denoise_wavelet(track['trail_x'], sigma=sigma, **config)
    y = denoise_wavelet(track['trail_y'], sigma=sigma, **config)
    track['trail_x'] = x
    track['trail_y'] = y
    tmodel = pwlpartition.contract(np.stack([x,y]))
    trail_step_idx = tmodel.get_time()
    trail_step_idx[-1] -= 1 
    _fj.lin(track, trail_step_idx, xkey='trail_x', ykey='trail_y')
    track.trail_step_idx = trail_step_idx

    return track

    # track['trail_x'] = denoise_wavelet(track['trail_x'], sigma=sigma, **config)
    # track['trail_y'] = denoise_wavelet(track['trail_y'], sigma=sigma, **config)
    # return track

# denoise_top = denoise_track(subsets['top'][0], use_sigma) 
denoise_top = [denoise_track(track, use_sigma) for track in subsets['top']]

# %%
# plot one 
ltrack = denoise_top[0]
ax = new_local_plot_tracks([ltrack.cut_time(0,20), subsets['top'][0].cut_time(0,20)])
ax.legend(['wavelet', 'original'])

# %%
# so sigma estimate is correlated with velocity?
topdatalist = [twanalyse.observables([tr]) for tr in denoise_top]

meanvel = get_array(make_get('lvel.mean'), topdatalist)
with mpl.rc_context(mplstyle):
    fig, ax= plt.subplots(figsize=(6,4))
    ax.scatter(meanvel, toplist['est_sigma'])
    ax.set_xlabel('mean velocity')
    ax.set_ylabel(r'estimate $\sigma$')

corr = np.corrcoef(meanvel, toplist['est_sigma'])[0][1]
print(f'correlation {corr:.4f}' )
# so there is a positive correlation

# %%
# relinearse using 0.12  microns
linearise_top = [_fj.linearize(track, step_d=0.12) for track in denoise_top]
lintopdatalist = [twanalyse.observables([tr]) for tr in linearise_top]

# %%
# compare new and old deviation.var
oldtopdatalist = [fjlocal[idx] for idx in subset_idx['top']]
 
new_var = get_array(make_get('deviation.var'), lintopdatalist)
old_var = get_array(make_get('deviation.var'), oldtopdatalist)
    
def compare_distrib(ax, a, b):
    defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hstyle = {'stat':'density', 'alpha':0.4, 'element':'step'}
    sns.histplot(a, ax=ax, color=defcolor[0], **hstyle)
    sns.histplot(b, ax=ax, color=defcolor[1],  **hstyle)

fig , ax = plt.subplots(figsize=(6,4))
compare_distrib(ax, old_var, new_var)


# %%
# same for candidate
candidate = subsets['candidate'][0]
denoise_candidate = denoise_track(candidate, use_sigma)
lin_candidate = _fj.linearize(denoise_candidate, step_d=0.12)
new_candidate_summary = twanalyse.observables([lin_candidate])
fjlocal[2924]['deviation']['var'], new_candidate_summary['deviation']['var']

# partial_candidate_summary = twanalyse.observables([denoise_candidate])
# partial_candidate_summary['deviation']['var']

# so new and old deviation angles are much the same as expected after linearisation

# %%
# * linearised this new transform



