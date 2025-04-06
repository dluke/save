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
# Rejection ABC against fanjin.top and simulated target using vret as  variable instead of \alpha

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

# %% 
plotting = False

# %% 
# config
notename = 'vretabc'
warnings.filterwarnings("ignore")
plt.rcParams.update({
    'text.usetex': False,
    })


# %% 
# load fanjin 
all_idx, ltrs = _fj.slicehelper.load_linearized_trs("all")
reference_idx = _fj.load_subset_idx()
# %% 
objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
refdf = fjanalysis.compute_reference_data(ltrs, reference_idx, objectives)
subset = "top"
topref = refdf.loc[refdf['subset'] == subset]
topref

# %% 
# load sampling
sim4d = {}
sim4d["simdir"] = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d_vret/"
sim4d["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'fanjin.top.ks_statistic']
sim4d = abcimplement.load_problem_simulation(sim4d)
sim4d["problem"]
N = 50
print('{}/{}'.format( N, sim4d["M"]))



# %% 
sim4d["params"] = sim4d["data"].paramsdf(sim4d["objectives"])
abcimplement.transform_vret_parameter_data(sim4d)
statdf, statref = abcimplement.regularise_stats(sim4d["params"], topref, sim4d["objectives"])

# %% 
# compute  accepted
_objectives = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"]
_accepted = abcimplement.rejection_abc(statdf, _objectives, statref, N)
prime_accepted = _accepted


problem = sim4d["problem"]
df = abcimplement.tabulate_inference(problem, _accepted, _objectives)
m_par = {k : v for k, v  in zip(df["parameter"], df["MLE"])}
# m_par = {k : v for k, v  in zip(df["parameter"], df["weighted mean"])}
m_par
df

# %% 
lpar = [r'$\tau_{\mathrm{dwell}}$', r'$\kappa$', r'$v_{\mathrm{ret}}$', r'$k_{\mathrm{spawn}}$']
fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, mpar=m_par, lpar=lpar)
pub.save_figure('vret_crawling_abc', notename, fig, config={"svg":False})

# %% 

# %% 
# what happened to the scores?

prime_accepted.attrs
prime_accepted["score"]
prime_accepted
statref


# %% 
df = abcimplement.tabulate_inference(sim4d["problem"], prime_accepted, "fanjin top")
pub.pickle_df(df, "vret_bayes_inference", notename)
df

# %% 
# prediction of wide pili distribution is conserved wide 
m_par
# %% 
# then a typical retraction velocity is predicted to be
vret = m_par["kb_sh"] * 0.004
print('vret {:.4f} microns/s'.format(vret))
# which is about 5 times slower than expected
# how is is that spawn rate is still not localised?
# %% 
# dwell time and kspawn and heavily anticorrelated?
fig, axes = plt.subplots(figsize=(5,5))
sns.scatterplot(data=_accepted, x="dwell_time", y="k_spawn")
# apparently not ...

# %% 
# one objective at a time
accdata = {}
for objective in sim4d["objectives"]:
    _accepted = abcimplement.rejection_abc(statdf, [objective], statref, N)
    accdata[objective] = _accepted

# %% 
# plotting
for objective in accdata:
    _accepted = accdata[objective]
    fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, lpar=lpar)
    fig.suptitle(objective)

# %% 
# choose accepted
_accepted = accdata["lvel.mean"]

# %% 
# contractions
params = sim4d["params"]
params["contraction.mean"] = -1 * sim4d["data"].get("effective_contract_length.mean")
params["pdisp.mean"] = sim4d["data"].get("pdisp.mean")
params["npili"] = sim4d["data"].get("npili.mean")
params["nbound"] = sim4d["data"].get("nbound.mean")
params["ntaut"] = sim4d["data"].get("ntaut.mean")
acc_params = params.iloc[_accepted.index]
# %% 
sns.histplot(acc_params["ntaut"])
# %% 
fig, ax = plt.subplots(figsize=(10,5))
# sns.scatterplot(data=acc_params, x="k_spawn", y="npili")
sns.scatterplot(data=acc_params, x="k_spawn", y="dwell_time", hue="ntau")
# sns.scatterplot(data=acc_params, x="k_spawn", y="ntaut")
# %% 


fig, ax = plt.subplots(figsize=(5,5))
sns.scatterplot(data=acc_params, x='k_spawn', y='pdisp.mean')
fig, ax = plt.subplots(figsize=(5,5))
sns.scatterplot(data=acc_params, x='k_spawn', y='contraction.mean')

fig, ax = plt.subplots(figsize=(5,5))
sns.scatterplot(data=acc_params, x='dwell_time', y='pdisp.mean')
fig, ax = plt.subplots(figsize=(5,5))
sns.scatterplot(data=acc_params, x='dwell_time', y='contraction.mean')


# %% 
# load 2d reduced dimension sampling
sim2d = {}
sim2d["simdir"] = "/home/dan/usb_twitching/run/825bd8f/cluster/mc2d_vret/"
sim2d["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'fanjin.top.ks_statistic']
sim2d = abcimplement.load_problem_simulation(sim2d)
sim2d["problem"]
N = 200 
print('{}/{}'.format( N, sim2d["M"]))

# %% 
sim2d["params"] = sim2d["data"].paramsdf(sim2d["objectives"])
statdf, statref = abcimplement.regularise_stats(sim2d["params"], topref, sim2d["objectives"])
# %% 
# compute  accepted
_objectives = ["lvel.mean", "deviation.var", "ahat.estimate"]
_accepted = abcimplement.rejection_abc(statdf, _objectives, statref, N)
# m_par, v_par = abcimplement.mean_accepted(sim2d["problem"], _accepted)
# abcimplement.describe_abc(sim2d, _accepted)
# %% 
sim2d["problem"]

# %% 
# mpl.rcParams['axes.linewidth'] = 10
def accepted_plot2d(problem, accepted):
    _bounds = sim2d["problem"]["bounds"]
    fig, ax = plt.subplots(figsize=(5,5))
    sns.scatterplot(data=accepted, x="k_spawn",  y="kb_sh", hue="score")
    xb, yb = _bounds
    ax.set_xlim(yb)
    ax.set_ylim(xb)
    return fig, ax
fig, ax = accepted_plot2d(sim2d["problem"], _accepted)
ax.set_frame_on(True)
plt.title(_objectives)

# %% 
# one statistic at a time
for objective in sim2d["objectives"]:
    _acc = abcimplement.rejection_abc(statdf, [objective], statref, N)
    accepted_plot2d(sim2d["problem"], _acc)
    plt.title(objective)
# %%

_acc = abcimplement.rejection_abc(statdf, ['lvel.mean'], statref, N)
xdata = _acc["k_spawn"]
ydata = _acc["kb_sh"]

def candf(x, a, b, c):
    # return np.exp(-b*x)/(a*x) 
    # return a + b*x + c*x**2 + d*x**3 + e*x**4
    # return a * np.exp(-b * x) + c
    return 1/(a*x**c) 
p, pcov = scipy.optimize.curve_fit(candf, xdata, ydata)
print('p', p)
basis = np.linspace(0.01,1.5,1500)
fit = [candf(x, *p) for x in basis]

fig, ax = accepted_plot2d(sim2d["problem"], _acc)

basis = np.linspace(0.01,8.0,1500)
fit = [candf(x, *p) for x in basis]
ax.plot(basis, fit)
ax.set_xlim(0, np.max(xdata))
ax.set_ylim(0, np.max(ydata))
