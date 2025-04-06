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
# plot the curve coordinate (mapped time)
# we want to know if we can see and define "tumbles" in the mapped time

# %% 
import os
import random
import numpy as np
import pickle
join = os.path.join 
norm = np.linalg.norm
pi = np.pi

import scipy
import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
import pili

import _fj
print("writing figures to", pub.writedir)

from pili import support
import mdl
import annealing 
from synthetic import *

# %% 
# load candidate track
candidate_idx = 2924
original_tr = _fj.trackload_original([candidate_idx])[0]
lptr = mdl.get_lptrack(original_tr)
_T = 20
_data = lptr.cut(0,_T)
r = 0.03
# %% 
_guess = mdl.recursive_coarsen(_data, 10, parameter='M')
ax = plt.subplots(figsize=(20, 8))
mdl.plot_model_on_data(plt.gca(), _guess, _data)


# %% 
# plot initial guess with outliers colored
tmp_anneal = annealing.Anneal(r)
tmp_anneal.initialise(_guess, _data)
is_outlier = tmp_anneal.get_outliers()
fig, ax = plt.subplots(figsize=(20, 8))
mdl.plot_model_on_data(ax, _guess, _data,
        intermediate={'is_outlier':is_outlier}, config={"h_outlier":True})

# %% 
annealing.clear_logs()
def solve_exp_data(data, M, r):
    config = {"use_ordering": True}
    rngstate = np.random.RandomState(0)
    _guess = mdl.recursive_coarsen(data, M, parameter='M')
    anneal = annealing.Anneal(r)
    anneal.initialise(_guess, data)
    solver = annealing.Solver(anneal, rng=rngstate, config=config)
    try:
        solver.multiple_linear_solve(n=1, n_local=2)
        solver.random_queue_solve()
    except Exception as e:
        print(e)
        raise(e)
    return solver

solver = solve_exp_data(_data, 11, r)
print("finished")
# %% [markdown]
# Note: whether or not we use the ordering term for the random_queue_solve
# the 6th segment is out of place. We need to forcible associate the outlier with this segment and accept/reject

# %%
print(solver.anneal.llmodel)
interest = solver.anneal.seglist.closest_segment == 6
np.argwhere(interest)
interest_pt_index = 128

# %%

def local_plot(anneal, data, _config={"h_outlier":True}):
    # note specific plotting function
    fig, ax = plt.subplots(figsize=(20,20))
    is_outlier = anneal.get_outliers()
    mdl.plot_model_on_data(ax, anneal.get_current_model(), data, 
        intermediate={'is_outlier':is_outlier}, config=_config)

# local_plot(solver.anneal, _data)
local_plot(solver.anneal, _data, _config={"h_outlier":True, "h_points": [interest_pt_index]})

# %%
pwl, mapped = solver.anneal.get_pwl_coord(get_mapped=True)
fig, ax = plt.subplots(figsize=(20,20))
mdl.plot_mapped(ax, mapped, _data, solver.anneal.get_current_model())
pub.save_figure("experimental_plot_mapped", "mdlpwlcoord", config={"svg":True})
# %%
# comute the distance distribution
points = _data.get_n2()
_dist = np.linalg.norm((mapped - points), axis=1)
# !tmp sign calculation
sign =  np.empty(200)
for i in range(sign.size):
    sign[i] = 1.0 if mapped[i][1] > points[i][1] else -1.0
# ax = sns.histplot(_dist)
_xlim = (-0.07, 0.07)
ax = sns.histplot(sign * _dist, binrange=_xlim)
from scipy.stats import norm
loc, scale = norm.fit(sign * _dist)
basis = np.linspace(_xlim[0], _xlim[1], 1000)
pdf = [scipy.stats.norm.pdf(x, loc=loc, scale=scale) for x in basis]
ax.plot(basis, pdf, linestyle='--', c='k', linewidth="4")
ax.set_xlabel("model deviation ($\mu m$)")

# 
# %%
# As an experiment, remap the point of interest and solve with n_local = 0
copy_anneal = solver.anneal.clone() # %%
copy_anneal.seglist.remap = {128 : 6}
nm_options = {"fatol":0.01, "xatol":0.01}
node = copy_anneal.get_segment_node(6)
copy_anneal.local_lsq_optimise([node], n_local=0, options=nm_options, use_ordering=True)
# %%
local_plot(copy_anneal, _data, _config={"h_outlier":True, "h_points": [interest_pt_index]})

# %%
fig, ax = plt.subplots(figsize=(20,20))
mdl.plot_model_on_data(ax, None, _data)

# %%
pwlt = solver.anneal.get_pwl_coord()
ax = sns.histplot(pwlt, bins=50)
ax.set_xlabel("curve coordinate ($\mu m$)")
ax.set_ylabel("count datapoints")

# %%
sns.kdeplot(pwlt,bw=0.03)

# %%
pwlv = pwlt[1:] - pwlt[:-1]
fig, ax = plt.subplots(figsize=(16,8))
ax.plot(pwlv, c='k', alpha=0.5)
ax.set_xlabel("time")
ax.set_ylabel("pwl velocity")
# from skimage.restoration import denoise_wavelet
# pwlv_denoise = denoise_wavelet(pwlv, method='BayesShrink', mode='soft', wavelet='db1', sigma=0.02)
# ax.plot(pwlv_denoise, c='orange')


