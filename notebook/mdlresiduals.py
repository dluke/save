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
# analyse the residuals of the PWL model


# %% 
import os
import json
import numpy as np
import scipy.stats
import pickle
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili
from pili import support
import _fj
import mdl
import pwlpartition
import pwltree

import fjanalysis
import pwlstats

from skimage.restoration import denoise_wavelet
from skimage.restoration import  estimate_sigma

# %% 
mplstyle = {"font.size": 20}
notename = "mdlresiduals"

# %% 
# load the candidate PWL model
path = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
solver = pwlstats.load_solver_at(path)
partition = solver.partition
model = partition.model




# %% 

breakpts = np.insert(np.cumsum(model.get_step_length()),0,0)
coord = partition.get_curve_coord()
residuals = partition.residual_at(0, partition.model.M, inter_term=False, signed=True)

def new_plot_residuals(coord, residuals, lims=(0,10)):
    fig,ax= plt.subplots(figsize=(12,4))
    plot_residuals(ax,  coord, residuals, lims)
    
def plot_residuals(ax, coord, residuals, lims=(0,10)):
    ax.plot(coord, residuals, linestyle='none', marker='o', markersize='4', label="residuals")
    vstyle = dict(alpha=0.3, linestyle='-', color='orange')
    ax.axhline(0, color='k', alpha=0.4)
    for bpt in breakpts:
        ax.axvline(bpt, **vstyle)
    ax.axvline(0, label="breakpoint", **vstyle)
    ax.set_xlim(*lims)

    # # highlight
    # for i in range(len(coord)):
    #     ax.plot(coord[i], residuals[i])

edge = np.linspace(0,40,5)
for i in range(4):
    if False: # check near_idx calculation
        _coord = coord[~near_idx]
        _residuals = residuals[~near_idx]
    else:
        _coord = coord
        _residuals = residuals
    new_plot_residuals(_coord, _residuals, (edge[i],edge[i+1]))

# %%
with mpl.rc_context({'font.size': 16, "axes.labelsize": 20}):
    fig, ax = plt.subplots(figsize=(12,3))
    plot_residuals(ax, coord, residuals, lims=(0,8))
    ax.set_xlabel(r'curve distance $(\mu m)$')
    ax.set_ylabel(r'$\Delta x^\perp (\mu m)$', fontsize=24)
    ax.set_ylim((-0.05,0.05))
    ax.legend(loc=(1.04,0))

target = join(plot_target, "residuals_example.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)


# %%
# so the "unexplained" residual standard deviation is
solver.sigma, np.std(residuals)
sigma_unexp = np.std(residuals) - solver.sigma
print("{:.04f} - {:.04f} = {:.04f}".format(np.std(residuals), solver.sigma, sigma_unexp))

# %% [markdown]
# but its tricky to interpret this right away because of several biases
# * the original experimental error estimate is biased, we would rather have an empirical measure
# (also this estimate should be quite consistent across trajectories but it isn't)
# * we don't take into account the edge effects at corners of the PWL model, residuals should be larger here 

# %%
# lets check whether residuals are larger in the vicinity of breakpoints
# a sensible approach is to split our data using solver.r into points close to breakpoints and points far away
near_r = solver.sigma
near_r = solver.r

mapping = partition.get_mapping()
rdistance = np.abs(breakpts[mapping]-coord)
fdistance = np.abs(breakpts[mapping+1]-coord)
near_idx = np.logical_or(rdistance < near_r, fdistance < near_r)
np.mean(np.abs(residuals[near_idx])), np.mean(np.abs(residuals[~near_idx]))
# contrary to expectation, the residuals at the break points are slightly smaller

print("near breakpoint {}/{}".format(np.sum(near_idx),near_idx.size))

# %%
# TODO residual distribution is normal?
data = residuals
# data = residuals[near_idx]

# fig, axes = plt.subplots(3, 1, figsize=(15,5))


with mpl.rc_context({'font.size': 16}):
    fig, ax = plt.subplots(figsize=(4,4))
    # shstyle = dict(stat="density", element="step", fill=False, alpha=0.8)
    shstyle = dict(stat="density", alpha=0.8)
    sns.histplot(data , ax=ax, **shstyle)
    (mu, sigma) = scipy.stats.norm.fit(data)
    print(mu, sigma)
    xlim = (-0.1,0.1)
    ax.set_xlim(xlim)
    ax.set_xlabel(r'$\Delta x^\perp (\mu m)$')


    defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    normal_fit = scipy.stats.norm(mu, sigma)
    lims = data.min(), data.max() 
    space = np.linspace(*lims, num=1000)
    normal_fit.pdf(space)
    fitstyle = {"linestyle":"--", "linewidth":3, "alpha":0.8, "color" : defcolors[1]}
    ax.plot(space, normal_fit.pdf(space), **fitstyle)

target = join(plot_target, "residuals_distribution.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)



data.size, scipy.stats.shapiro(data)

# !residuals appear to be normally distrubted, by why? we know our data contains small non-linearities
# ! and the curve coordinates appear to have an exponetial error
# * least squares fitting doesn't bias residuals to be normally distributed


# %%
# why not just use pearson correlation coefficient?

for i in range(1,10):
    c = np.corrcoef(residuals[i:], residuals[:-i])[0][1]
    print(c)

# %%
# another approach is to calculate the correlation of the residuals
#
# https://www.statology.org/durbin-watson-test-python/
# https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic

from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(residuals))
# 1.43 indicates that the residuals positively correlated in the sense that 
# sequential residuals fall often on the same side of the trajectory
 
durbin_watson(residuals[near_idx]), durbin_watson(residuals[~near_idx])
# as expected this effect is less pronounced for points close to the breakpoints

# %%
# * LOAD more data
# run top dataset
select_idx = _fj.load_subset_idx()["top"]
look = [join(pwlstats.root, f"run/cluster/no_heuristic/top/_top_{idx:04d}")  for idx in select_idx]
found = [directory for directory in look if os.path.exists(join(directory, "solver.pkl"))]
solverlist= [pwlstats.load_solver_at(directory) for directory in found]
residual_list  = [solver.partition.get_signed_residuals() for solver in solverlist]

# %%
print(len(found))
dwstat = [durbin_watson(residual) for residual in residual_list]

with mpl.rc_context({'font.size': 16}):
    fig,ax = plt.subplots(figsize=(6,4))
    sns.histplot(dwstat,ax=ax)
    ax.set_xlim(0,4)
    ax.set_xlabel("Durbin-Watson statistic")
    ax.axvline(2.0,linestyle='--', color='k', alpha=0.4)


target = join(plot_target, "population_durbin_watson.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)


# consistent results on the population datak
# %%

fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(np.diff(coord))
ax.set_xlim((None,0.2))




# %%
