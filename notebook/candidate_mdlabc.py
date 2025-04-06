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
# use our linear regression analysis of the candidate trajectory to calibrate it against simulation sampling

# %% 
import warnings
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
import command
import stats
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
from pili import support
import pwlstats
import pwlpartition

# %% 
plotting = True

# %% 
# config
notename = 'candidate_mdlabc'
warnings.filterwarnings("ignore")
plt.rcParams.update({
    'text.usetex': False,
    })



# %% 
# load fanjin 

target = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")

solver = pwlpartition.Solver.load_state(join(target, "solver"))

# !tmp
solver.partition.use_probability_loss = False
solver.partition.inter_term = False
# !~tmp


config = support.load_json(join(target, 'config.json'))
track = _fj.trackload_original([config["trackidx"]])[0]
lintrack = _fj.lintrackload([config["trackidx"]])[0]

candidate_summary = pwlstats.solver_summary(solver, track)
target_msl = candidate_summary["median_step_length"]
target_dstd = np.var(candidate_summary["deviation"])
print(f'candidate median displacement length is {target_msl:.4f} mircons')
print(f'candidate deviation angle var {target_dstd:.4f} ')



# %% 
objectives = ['lvel.mean', 'deviation.var', 'pdisp.median']
# ref = [target_msl]

refdf = fjanalysis.compute_reference_data([lintrack], {"candidate": [0]}, objectives)
refdf['pdisp.median'] = target_msl
refdf['deviation.std'] = target_dstd
reference = refdf
reference

# %% 
# load sampling
sim4d = {}
sim4d["simdir"] = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d_vret/"
sim4d["objectives"] = objectives
sim4d = abcimplement.load_problem_simulation(sim4d)
sim4d["problem"]
# N = 200 
N = 100
print('{}/{}'.format( N, sim4d["M"]))

# %%
# * ------------------------------------------------------------
# practice loading distribution data
# how long does it take?
abc_target = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d_vret"
distrib_path = "xydisp.npy"
def load_abc_distrib(target, path, lookup):
    datalist = []
    for udir in lookup[0]:
        npypath = join(target, udir, path)
        data = np.load(npypath)
        datalist.append(data)
    return datalist

with support.Timer():
    datalist = load_abc_distrib(abc_target, distrib_path, sim4d['lookup'])

# check that all simulations return some pdisp data
count_pdisp = np.array(list((map(len, datalist))))
fig, axes = plt.subplots(1, 2, figsize=(10,4))
ax = axes[0]
sns.histplot(count_pdisp, ax=ax)
ax.set_xlabel('entries')
count_pdisp.min(), count_pdisp.max()

mean_pdisp = np.array(list((map(np.mean, datalist))))
ax = axes[1]
ax.set_xlabel('mean pdisp')
sns.histplot(mean_pdisp, ax=ax)

# %% 
# %% 
# %% 
# construct a new statistic by comparing the step length and pili displacement distributions
sigma = candidate_summary['estimate_sigma']
r = solver.r
ref_distrib = candidate_summary["lengths"]
def pdisp_ks_statistic(datalist, ref_distrib, r):
    threshold = 2*r
    def ks_stat(sim, ref):
        sim = sim[sim>threshold]
        ks_statistic, pvalue = scipy.stats.ks_2samp(sim, ref)
        return ks_statistic
    return np.array([ks_stat(sim_distrib, ref_distrib) for sim_distrib in datalist])
    
with support.Timer():
    pdks = pdisp_ks_statistic(datalist, ref_distrib, r)
pdks

# %% 
# pdisp.median but adjusted by threshold
median_pdisp = np.array([np.median(sim) for sim in datalist])
adjusted_median_pdisp = np.array([np.median(sim[sim>2*r]) for sim in datalist])

fig , ax = plt.subplots(figsize=(6,4))

def compare_distrib(ax, a, b):
    defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hstyle = {'stat':'density', 'alpha':0.4, 'element':'step'}
    sns.histplot(a, ax=ax, color=defcolor[0], **hstyle)
    sns.histplot(b, ax=ax, color=defcolor[1],  **hstyle)

compare_distrib(ax, median_pdisp, adjusted_median_pdisp)
ax.set_xlabel(r'length $\mu m$')
ax.legend(['pdisp.median', 'adjusted'])

# %% 
# retrieve the best simulation pdisp distribution
simidx = pdks.argmin()
simuid = sim4d['lookup'][0][simidx]
simuid
best_pdks = datalist[simidx]
best_pdks = best_pdks[best_pdks > 2*r]
# %% 
# retrieve the best simulation pdisp distribution
# best simultion is _u_fsCKYXbI
# plot them side by side

fig , ax = plt.subplots(figsize=(6,4))
compare_distrib(ax, ref_distrib, best_pdks)
ax.set_xlabel(r'length $\mu m$')
ax.legend(['candidate', simuid[3:]])


# %% 
# sim4d['lduid']['_u_UhuVf4Rk']['fanjin']['candidate']['ks_statistic']
# todo : add the ks_statistic and adjusted mean/median to the parameter dataframe and then do abc with these statistics
# todo : compute fast linear models for the entire dataset
# todo : analyse the trailing pole

# * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% 

params = sim4d["data"].paramsdf(objectives)
sim4d["params"] = params

params['pdks'] = pdks
params['adjusted_median_pdisp'] = adjusted_median_pdisp
reference['pdks'] = 0
reference['adjusted_median_pdisp'] = reference['pdisp.median']

reference

# %% 
abcimplement.transform_vret_parameter_data(sim4d)
statdf, statref = abcimplement.regularise_stats(sim4d["params"], reference, sim4d["objectives"])

statdf.attrs['std']
statdf

# %% 
# compute  accepted
_objectives = ['lvel.mean', 'deviation.var', 'pdks']
# _objectives = ['lvel.mean', 'pdks']
# _objectives = ['pdks']
_accepted = abcimplement.rejection_abc(statdf, _objectives, statref, N)
prime_accepted = _accepted
m_par, v_par = abcimplement.mean_accepted(sim4d["problem"], _accepted)
abcimplement.describe_abc(sim4d, _accepted)

# %% 
lpar = [r'$\tau_{\mathrm{dwell}}$', r'$\kappa$', r'$v_{\mathrm{ret}}$', r'$k_{\mathrm{spawn}}$']
fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, mpar=m_par, lpar=lpar)
# pub.save_figure('vret_crawling_abc', notename, fig, config={"svg":False})

# %% [markdown]
# we need to run the PWL solver on simulated  data!
# this is easier said than done since the simulated data is much larger than experimental
# one shortcut is to simply discard all the pdisp.xydisplacements that are below the estimated noise threshold
# the proper way to proceed is to use our true knowledge of the simulated data to generate a nealy ideal initial guess?
