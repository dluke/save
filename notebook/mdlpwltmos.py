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
# examine simulated data again but this time 
# compare greedy pwl solve of the simulated data against the best equivalent candidate pwl solve
# extending ideas in mdltmos.py

# %%
import os
import json
import numpy as np
import scipy.stats
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd

import readtrack
import parameters
import _fj

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

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)


# %%
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
notename = 'mdlpwltmos'
mplstyle = {"font.size": 20}

# %%
# load simulation pwl summary for one simulation to start with
simid = 'HH161KlZ'
target = join(pili.root, '../run/825bd8f/cluster/mc4d_vret/_u_HH161KlZ')

lsummary, tsummary = pwlstats.sim_summary(target)
def describe_sim_summary(summary):
    N = summary['N']
    duration = summary['duration']
    contour_length = summary['contour_length']
    print(f'simulated pwl solve has {N} segments for {duration}s of trajectory data')
    print(f'total contour length {contour_length:.2f} microns')
describe_sim_summary(lsummary)

# %%

simdir = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d_vret/"
lookup = sobol.read_lookup(simdir)
problem = sobol.read_problem(simdir)
lduid = sobol.collect_lduid(simdir)
problem

# %%
with support.Timer():
    summary = lambda uid : pwlstats.sim_summary(join(simdir, uid))
    datalist = []
    for uid in lookup[0]:
        data = summary(uid)
        if data is None:
            continue
        datalist.append((uid, data))

print('found {}/{}'.format(len(datalist), len(lookup[0])))
# %%
valid, ltdata = zip(*datalist)
ldata, tdata = zip(*ltdata)

# %%
# compile the model parameters for valid data
def compile_params(problem, lookup, valid):
    _parlist = list(zip(problem["names"],  zip(*[lookup[1][_u] for _u in valid])))
    _col = {}
    _col["uid"] = [_u  for _u in valid]
    _col.update({k:v for k, v in _parlist})
    return pd.DataFrame(_col)
params = compile_params(problem, lookup, valid)

# %%

lduidlist = [lduid[_u] for _u in valid]
dev_var = get_array(make_get('deviation.var'), lduidlist)
print('number of failed simulations by deviation.var', np.sum(np.isnan(dev_var)))
dev_var[np.isnan(dev_var)] = 0
# old wavelet statistics
params['deviation.var'] = dev_var 
params

# %%
from support import make_get, get_array
mean_velocity = get_array(make_get('contour_length'), ldata) / get_array(make_get('duration'), ldata)
step_rate = get_array(make_get('N'), ldata) / get_array(make_get('duration'), ldata)
median_step = [np.median(ld['lengths']) for ld in ldata]

# %%
# fig, axes = plt.subplots(1,3,figsize=(12,4))
params['vel'] = mean_velocity
params['step_rate'] = step_rate
params['step.median'] = median_step
params

# %%
import fjanalysis
old_deviation_var =  fjanalysis.load_summary()[2924]['deviation']['var']

# %%
# compute the reference data
target = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
candidate_summary = pwlstats.load_candidate_statistics(target)
_DT = 0.1
candidate_summary["duration"] *= _DT
reference = pd.DataFrame(
    {
        'vel': candidate_summary["contour_length"]/candidate_summary["duration"],
        'step_rate': candidate_summary["N"]/candidate_summary["duration"],
        'step.median': candidate_summary["median_step_length"]
    },
    index = ['candidate']
)
reference['deviation.var'] = old_deviation_var
reference

# %%
# ks statistics
ref = candidate_summary["lengths"]
ks_score = lambda x: scipy.stats.ks_2samp(x, ref)[0]
ks_pwl_length = [ks_score(ld['lengths']) for ld in ldata]
params['ks_length'] = ks_pwl_length

# %%
# finally approximate bayesian computation
# objectives = ['vel', 'step_rate', 'step.median']
# objectives = ['vel', 'step_rate', 'ks_length'] # !best
objectives = ['vel', 'step_rate', 'ks_length', 'deviation.var']
# objectives = ['ks_length']

statdf, statref = abcimplement.regularise_stats(params, reference, objectives)
# statdf.attrs
# statdf, statref = params, reference

# %%
N = 50
_accepted = abcimplement.rejection_abc(statdf, objectives, statref, N, norm=2)
m_par, v_par = abcimplement.mean_accepted(problem, _accepted)

# %%
sort_accepted = statdf.sort_values("score")[:50]
sort_score = sort_accepted['score'].to_numpy()

# print('epsilon = ', sort_accepted['score'][-1])
print('best, epsilon', sort_score[0], sort_score[-1])
sort_accepted[:10]


# %%
lpar = [r'$\tau_{\mathrm{dwell}}$', r'$\kappa$', r'$v_{\mathrm{ret}}$', r'$k_{\mathrm{spawn}}$']
fig, axes = abcimplement.perfectplot4d(problem, _accepted, mpar=m_par, lpar=lpar)

# pub.save_figure('candidate_abc_vel_steprate_kslength_devvar.png', notename)

# %% 
# %% 
sort_params = params.iloc[sort_accepted.index][:10]
sort_params
# %%
reference

# %%

lduid[sort_params['uid'].iloc[0]]
