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
# attempt to infer simulation parameters from simulated data using pwl statistics

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


from pili.support import make_get, get_array

# %%
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
notename = 'sim_pwl_vretabc'
mplstyle = {"font.size": 20}

# %%
# SETUP OBJECTIVES

# candidate target
target_name = 't6'
# top target
# target_name = 't7' 

objectives = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate", "kmsd.mean"]
# objectives = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"]
objectives.append(f"sim.{target_name}.ks_statistic")
objectives.append(f"fanjin.candidate.ks_statistic")


# %%
# ----------------------------------------------------------------
# SIMULATED REFERENCE
target = join(pili.root, '../run/825bd8f/target/', target_name)
print('target', target)
args = parameters.thisread(directory=target)
lrefdata, trefdata = pwlstats.sim_summary(target)
lrefdata
summary = lrefdata

reference = pd.DataFrame(
    {
        'vel': summary["contour_length"]/summary["duration"],
        'step_rate': summary["N"]/summary["duration"],
        'step.median': np.median(summary["lengths"]),
        'angle.corrcoef': summary["angle"]["corrcoef"][0]
    },
    index = ['simtarget']
)


with command.chdir(target):
    local = stats.load()
    for obs in objectives:
        reference[obs] = make_get(obs)(local)
reference


# %%
simdir = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d_vret/"
lookup = sobol.read_lookup(simdir)
problem = sobol.read_problem(simdir)
lduid = sobol.collect_lduid(simdir)
simpar = {name : args.pget(name) for name in problem["names"]}
problem

# %%

def load_abc_pwl(simdir):
    summary = lambda uid : pwlstats.sim_summary(join(simdir, uid))
    datalist = []
    for uid in lookup[0]:
        data = summary(uid)
        if data is None:
            continue
        datalist.append((uid, data))

    print('found {}/{}'.format(len(datalist), len(lookup[0])))
    valid, ltdata = zip(*datalist)
    ldata, tdata = zip(*ltdata)
    return valid, ldata, tdata

# * 96s
with support.Timer():
    valid, ldata, tdata = load_abc_pwl(simdir)


# %%
def compile_params(problem, lookup, valid):
    _parlist = list(zip(problem["names"],  zip(*[lookup[1][_u] for _u in valid])))
    _col = {}
    _col["uid"] = [_u  for _u in valid]
    _col.update({k:v for k, v in _parlist})
    return pd.DataFrame(_col)
params = compile_params(problem, lookup, valid)

# %%
# * note we used velocity autocorrelation function in the past but its omitted from the current analysis
# * we compute kmsd.mean but run/825bd8f is not new enough to have it

Ym = sobol.collect(objectives, targetdir=simdir)
nans = {}
for name in Ym.keys():
    nan = np.isnan(Ym[name])
    if any(nan):
        print("nan found in {}. filtering {} samples".format(name, int(np.sum(nan))))
    nans[name] = nan
invalid = np.logical_and.reduce(list(nans.values()))
print('number of failed simulations', np.sum(invalid))

# %%
lduidlist = [lduid[_u] for _u in valid]
for obs in objectives:
    var = get_array(make_get(obs), lduidlist)
    params[obs] = var
params

# %%
# compile the objectives
def compile_objectives(params, ldata, lrefdata):
    mean_velocity = get_array(make_get('contour_length'), ldata) / get_array(make_get('duration'), ldata)
    step_rate = get_array(make_get('N'), ldata) / get_array(make_get('duration'), ldata)
    median_step = [np.median(ld['lengths']) for ld in ldata]

    params['vel'] = mean_velocity
    params['step_rate'] = step_rate
    params['step.median'] = median_step
    params['angle.corrcoef'] = get_array(lambda ld: make_get('angle.corrcoef')(ld)[0], ldata)

    # ks statistics
    ref = lrefdata["lengths"]
    ks_score = lambda x: scipy.stats.ks_2samp(x, ref)[0]
    ks_pwl_length = [ks_score(ld['lengths']) for ld in ldata]
    params['ks_length'] = ks_pwl_length

    ref = lrefdata["angles"]
    ks_score = lambda x: scipy.stats.ks_2samp(x, ref)[0]
    ks_pwl_angle = [ks_score(ld['angles']) for ld in ldata]
    params['ks_angle'] = ks_pwl_angle

# this takes 40s ( more likely several minutes? )
with support.Timer() as t:
    compile_objectives(params, ldata, lrefdata)

params

# %%
# ! compare to experiment
# reference = fj_reference
# ! compare to simulation
reference = reference

objectives = ['vel', 'step_rate', 'ks_length']
objectives = ['vel', 'step_rate', 'ks_length', 'ks_angle']
objectives = ['vel', 'step_rate', 'ks_length', 'ks_angle', 'deviation.var']
objectives = ['vel', 'step_rate', 'ks_length', 'deviation.var']

objectives = ['vel', 'ks_length', 'deviation.var', 'ahat.estimate']
objectives = ['vel', 'ks_length', 'deviation.var']
objectives = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"]

objectives = [f"sim.{target_name}.ks_statistic"]
objectives = [f"sim.{target_name}.ks_statistic", "deviation.var", "ks_length"]
# objectives = [f"sim.{target_name}.ks_statistic", "deviation.var", "qhat.estimate", "ahat.estimate"]
# objectives = [f"fanjin.candidate.ks_statistic", "deviation.var", "qhat.estimate", "ahat.estimate"]

# objectives = ['ks_length']
# objectives = ["vel", "fanjin.candidate.ks_statistic", "deviation.var"]

objectives = [f"sim.{target_name}.ks_statistic", "deviation.var", "step_rate"]
# objectives = ["vel", "", "step_rate"]
objectives = [f"sim.{target_name}.ks_statistic"]
objectives = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"]

statdf, statref = abcimplement.regularise_stats(params, reference, objectives)

# %%
N = 50
use_mle = True

_accepted = abcimplement.rejection_abc(statdf, objectives, statref, N, norm=2)
# _accepted = abcimplement.llregression(_accepted, objectives, statref, problem["names"])

# m_par, v_par = abcimplement.mean_accepted(problem, _accepted)
df = abcimplement.tabulate_inference(problem, _accepted, objectives)
print(simpar)
m_par = {k : v for k, v  in zip(df["parameter"], df["MLE"])}
# m_par = {k : v for k, v  in zip(df["parameter"], df["weighted mean"])}
m_par
df

# %%
lpar = [r'$\tau_{\mathrm{dwell}}$', r'$\kappa$', r'$v_{\mathrm{ret}}$', r'$k_{\mathrm{spawn}}$']
fig, axes = abcimplement.perfectplot4d(problem, _accepted, mpar=m_par, simpar=simpar, lpar=lpar)

# pub.save_figure('best_sim_pwl_candidate_vretabc', notename)

# %%
sort_accepted = statdf.sort_values("score")[:50]
sort_score = sort_accepted['score'].to_numpy()

# print('epsilon = ', sort_accepted['score'][-1])
print('best, epsilon', sort_score[0], sort_score[-1])
params.iloc[sort_accepted.index][:10]

# %%
reference
# %%
df, sc = abcimplement.tabulate_inference(problem, _accepted, objectives, simpar=simpar, extra=True, use_mle=use_mle)
# sc
df

# %%
#  the rows of our parameter inference quality table
obs_setup = [
    # ["kmsd.mean"],
    ["angle.corrcoef"],
    [f"sim.{target_name}.ks_statistic"],
    ["lvel.mean"],
    ["deviation.var"],
    ["qhat.estimate"],
    ["ahat.estimate"],
    ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"],
    ["lvel.mean", "deviation.var", "ahat.estimate"],
    ["lvel.mean", "deviation.var", "ahat.estimate", "angle.corrcoef"],
    ['vel'], 
    ['step_rate'],
    ['step.median'],
    ['ks_length'],
    ['ks_angle'],
    ['vel', 'step_rate', 'step.median'],
    ['vel', 'step_rate', 'ks_length'],
    ['vel', 'ks_length', 'deviation.var'],
    ['vel', 'ks_length', 'deviation.var', 'ahat.estimate'],
    [f"sim.{target_name}.ks_statistic"],
    [f"sim.{target_name}.ks_statistic", "deviation.var"],
    [f"sim.{target_name}.ks_statistic", "deviation.var", "ks_length"],
    [f"sim.{target_name}.ks_statistic", "deviation.var", "qhat.estimate", "ahat.estimate"]
]

def rejection_abc_inference(problem, params, reference, objectives, simpar, N=50, use_mle=use_mle):
    statdf, statref = abcimplement.regularise_stats(params, reference, objectives)
    _accepted = abcimplement.rejection_abc(statdf, objectives, statref, N, norm=2)
    df, sc = abcimplement.tabulate_inference(problem, _accepted, objectives, simpar=simpar, extra=True)
    return df, sc

rows = []
for objectives in obs_setup:
    df, sc = rejection_abc_inference(problem, params, reference, objectives, simpar, N=50)
    rows.append(sc)
rows[0]

# %%
from IPython.display import display, HTML
qdf = pd.concat(rows, ignore_index=True)
pub.save_dfhtml(qdf, f"sim_{target_name}_vretabc", notename)
qdf
# display(HTML(qdf.to_html()))

# %%
# compare xydisp and length distribution for reference simulation
path =  join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
sigma, r = pwlstats.load_candidate_sigma_r(path)

xydisp = np.load(join(target, 'xydisp.npy'))
_xydisp = xydisp[xydisp != 0.0]
with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(6,4))
    support.compare_distrib(ax, _xydisp, lrefdata["lengths"])
    ax.legend(['pdisp', 'pwl length'])
    ax.axvline(2*r,  c='k', linestyle='--', alpha=0.5)
    ax.set_xlim(0,1.4)
    ax.set_xlabel('microns')
pub.save_figure('compare_pdisp_pwl_'+target_name, notename)

# %%
# ------------------------------------------------------------------
# t6 only
best_uid = sort_accepted['uid'].iloc[0]
target_best = join(simdir, best_uid)

lvel = np.load(join(target_best, 'lvel.npy'))
vret = simpar['kb_sh'] * 0.004

fig, ax = plt.subplots(figsize=(6,4))
xlim = (0, 0.8)
sns.histplot(lvel, ax=ax, stat='density', element="step", fill=False, alpha=0.8, binrange=xlim)
ax.set_xlabel('step velocity')

# candidate_lvel = _fj.load_subsets()['candidate'][0].get_step_speed()
# c_summary = fjanalysis.load_summary()[2924]

fj_lvel = np.concatenate([tr.get_step_speed() for tr in _fj.load_subsets()['top']])


sns.histplot(fj_lvel, ax=ax, stat='density', element="step", fill=False, alpha=0.8, binrange=xlim)

# ax.legend(['t6', 'candidate'])
ax.legend([best_uid[3:], 'candidate'])
ax.legend([best_uid[3:], 'top'])

# %%
# check the nbound/ntaut
print('nbound', 'ntaut')
print(make_get('nbound.mean')(lduid[best_uid]), make_get('ntaut.mean')(lduid[best_uid]))
print('deviation.var',  make_get('deviation.var')(lduid[best_uid]))
reference

# %%
# ------------------------------------------------------------------
# t7 only
# now look at the lvel distributions for the reference and accepted samples at high and low kspawn
low_kspawn_uid = '_u_eEftbHmu'
high_kspawn_uid =  '_u_Zc9BWzBu'

lvel = np.load(join(target, 'lvel.npy'))
vret = simpar['kb_sh'] * 0.004

fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(lvel, ax=ax, stat='density', element="step", fill=False, alpha=0.8)
ax.axvline(vret,  c='k', linestyle='--', alpha=0.5)
ax.set_xlabel('step velocity')
ax.axvline(reference['lvel.mean'][0],  c='b', linestyle='--', alpha=0.5)

alpha=0.6

lklvel = np.load(join(simdir, low_kspawn_uid, 'lvel.npy'))
sns.histplot(lklvel, ax=ax, stat='density', element="step", fill=False, alpha=alpha)

hklvel = np.load(join(simdir, high_kspawn_uid, 'lvel.npy'))
sns.histplot(hklvel, ax=ax, stat='density', element="step", fill=False, alpha=alpha)


ax.legend(['lvel', 'reference', 'lvel.mean', low_kspawn_uid[3:]+' (low)', high_kspawn_uid[3:]+' (high)'])

row1 = params.loc[params['uid'] == low_kspawn_uid]
row2 = params.loc[params['uid'] == high_kspawn_uid]
simpar['k_spawn'], row1['k_spawn'].iloc[0], row2['k_spawn'].iloc[0], 

pub.save_figure("compare_lvel_distrib_vary_kspawn_simtarget_t7", notename)



# %%
