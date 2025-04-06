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
# Rejection ABC against a simulated target

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
import thesis.publication as thesis

# %% 
plotting = False

# %% 
# config
plt.rcParams.update({
	'text.usetex': False,
	'figure.figsize': (20,20),
	'axes.labelsize': 16
	})
notename = 'simabc'
verbose = False


# %% 
simtarget = "/home/dan/usb_twitching/run/825bd8f/target/t0"
with command.chdir(simtarget):
	ltarget = stats.load()
	args = parameters.thisread()
_simobjective = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'kmsd.mean',
	'double_ltrs.qhat.estimate', 'double_ltrs.ahat.estimate', 
	'quad_ltrs.qhat.estimate', 'quad_ltrs.ahat.estimate', 
	'cell_ltrs.qhat.estimate', 'cell_ltrs.ahat.estimate']
simref = {name : twutils.make_get(name)(ltarget) for name in _simobjective}
_interest = ['dwell_time', 'k_spawn', 'pilivar',  'anchor_angle_smoothing_fraction']
# print(parameters.describe(args, target=_interest))
simpar = {par : args.pget(par) for par in _interest}
simpar['anchor_angle_smoothing_fraction'] *= np.pi/2
simpar, simref

# %% 
sim4d = {}
sim4d["simdir"] = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d"
sim4d["objectives"] = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate', 'fanjin.top.ks_statistic', 'kmsd.mean']
sim4d = abcimplement.load_problem_simulation(sim4d)
sim4d["problem"]

# %% 
# ABC config
N = 200 
print('{}/{}'.format( N, sim4d["M"]))

# %%
# one statistic at a time
_objectives = _simobjective 
sim4d["params"] = sim4d["data"].paramsdf(_objectives)
abcimplement.transform_anchor_parameter_data(sim4d)
statdf, statref = abcimplement.regularise_stats(sim4d["params"], simref, _objectives)
#
statref
# %%
special_stat = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
spar = [r'$\langle u \rangle$', r'$Var(\theta_D)$', r'$\hat{q}$', r'$\hat{a}$']
_pretty = dict([(a,b) for a, b in zip(special_stat, spar)])
special = True
# TMP
_objectives = ['qhat.estimate', 'ahat.estimate',
	'double_ltrs.qhat.estimate', 'double_ltrs.ahat.estimate', 
	'quad_ltrs.qhat.estimate', 'quad_ltrs.ahat.estimate', 
	'cell_ltrs.qhat.estimate', 'cell_ltrs.ahat.estimate'
	]
_titles = [
	'persistence, step = 0.12', 'activity, step = 0.12',
	'persistence, step = 0.24', 'activity, step = 0.24',
	'persistence, step = 0.48', 'activity, step = 0.48',
	'persistence, step = 1.00', 'activity, step = 1.00',
]
# ~TMP
if plotting or special:
	for i, objective in enumerate(_objectives):
		_regdf = statdf[sim4d["problem"]["names"] + [objective]]
		_accepted = abcimplement.rejection_abc(_regdf, [objective], statref, N)
		fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=simpar)
		fig.suptitle(_titles[i], fontsize=40)
		if objective in special_stat:
			pass
			plt.savefig('jure/sim_crawling_abc_statistic_{}.png'.format(objective))
			# fig.suptitle(_pretty[objective])
			# pub.save_figure('sim_crawling_abc_statistic_{}'.format(objective), notename, fig, config={"svg":False})

# %% [markdown]
# because we have easy access to this data, we list the mean displacment-per-pilus 
# of the accepted samples for approximate bayesian computation using persistence
# and activity statistics with varying linearisation step sizes.

# %%
# FOR JURE
# compute mean per TFP displacement for each set of accepted samples
# _delta = sim4d["data"].get("effective_contract_length.mean")
_delta = sim4d["data"].get("pdisp.mean")
print('pdist.mean (all samples)', np.mean(_delta))
lst = []
for i, objective in enumerate(_objectives):
	_regdf = statdf[sim4d["problem"]["names"] + [objective]]
	_accepted = abcimplement.rejection_abc(_regdf, [objective], statref, N)
	acc_delta_l = _delta[_accepted.index]
	lst.append(np.mean(acc_delta_l))

r = [0.12,0.24,0.48,1.0]
a, b = lst[::2], lst[1::2]
df = pd.DataFrame({"step":r, "per TFP displacement (persistence)": a, "per TFP displacement (activity)" : b})
df

# %%
if plotting:
	_accepted = abcimplement.rejection_abc(statdf, ['kmsd.mean'], statref, N)
	fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=simpar)
	fig.suptitle(objective)


# %%
# pull the lvel statistic, k_spawn vs. \alpha relationship as a scatter plot


# %%
# all three simple metrics
warnings.filterwarnings("ignore")
_objectives = ["lvel.mean", "deviation.var", "ahat.estimate"]
_regdf = statdf[sim4d["problem"]["names"] + _objectives]
_accepted = abcimplement.rejection_abc(_regdf, _objectives, statref, N)
m_par, v_par = abcimplement.mean_accepted(sim4d["problem"], _accepted)
abcimplement.describe_abc(sim4d, _accepted)
prime_accepted = _accepted
# %%
# marginal standard deviations
df = abcimplement.tabulate_inference(sim4d["problem"], prime_accepted, "fanjin top", simpar=simpar)
pub.pickle_df(df, "bayes_inference", notename)
df

# %%

# where to put this utility?
def attr_lims(df, objectives):
	df.attrs['lims'] = {obs : (df[obs].min(), df[obs].max()) for obs in objectives}
	return df
unt_accepted = attr_lims(sim4d["params"].iloc[_accepted.index], _objectives)
unt_accepted.attrs['lims']

if True:
	fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=simpar, mpar=m_par)
	# sup = r'$ || \left( \mean{u} - \mean{u^prime}, Var({\theta_d}) - Var({\theta_d^\prime}), \hat{a} - \hat{a}^\prime \right) ||  < \epsilon $'
	# fig.suptitle(sup, fontsize=48, y=1.08)
	# pub.save_figure('crawling_rejectionabc', notename, fig, config={"svg":False})
	pub.save_figure('sim_crawling_abc', notename, fig)

# %%
print('pilivar', m_par['pilivar'])
print('anchor', m_par['anchor_angle_smoothing_fraction'])

# %%
# print 1d summary statistic distributions
_objectives = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"]
abcimplement.plot_accepted_stats(sim4d["params"], _accepted, _objectives, simref)

# %%
# delta l distribution 
delta_l = sim4d["data"].get("effective_contract_length.mean")
acc_delta_l = delta_l[_accepted.index]
fig, ax = plt.subplots(figsize=(5,5))
# ax = sns.histplot(delta_l, ax=ax)
ax = sns.histplot(acc_delta_l, ax=ax)
ax.set_xlabel(r"$\Delta l$")

# %%
# statistic v parameter correlation matrix for accepted samples
corrmat = np.zeros((4,4))
problem = sim4d['problem']
accobjective = ["lvel.mean", "deviation.var", "qhat.estimate", "ahat.estimate"]
_accparams  = sim4d["params"].iloc[_accepted.index]
for i, objective in enumerate(accobjective):
	for j, name in enumerate(problem['names']):
		_sacc = _accparams.sort_values(name)
		corr = np.corrcoef(_sacc[name], _sacc[objective])[0][1]
		corrmat[i,j] = corr
corrdf = pd.DataFrame(corrmat.T, index=problem['names'],  columns=accobjective)
corrdf

# %%
# we see correlations between persistence and dwell_time/pilivar even though 
# we fail to estimate dwell_time. 
# plotting
fig, axes = plt.subplots(1,2, figsize=(10,5))
ax = axes[0]
sns.scatterplot(data=_accparams, x='dwell_time', y='qhat.estimate', ax=ax)
ax = axes[1]
sns.scatterplot(data=_accparams, x='pilivar', y='qhat.estimate', ax=ax)


# %%
# load relevant 1d sampling
rundir = '/home/dan/usb_twitching/run/'
ttdwelldir = join(rundir, '825bd8f/target/t0_tdwell')
tanchordir = join(rundir, '825bd8f/target/t0_anchor')
dc_tdwell = rtw.DataCube(target=ttdwelldir)
dc_anchor = rtw.DataCube(target=tanchordir)
_ = dc_anchor.load_local()
_ = dc_tdwell.load_local()
# %%
# define a function to recover desired statistic
# mean contraction of a bound pilus
dlstat = 'effective_contract_length.mean'
# mean planar displacement associated with a single pilus
dispstat = 'pdisp.mean'
def _get_par(dc, var):
	lld = dc.load_local()
	contraction = [rtw.make_get(var)(ld) for ld in lld]
	basis = dc.basis[0]
	return np.array(basis), np.array(contraction)
tdwell, tdwell_contraction = _get_par(dc_tdwell, dlstat)
anchor, anchor_contraction = _get_par(dc_anchor, dlstat)
tdwell, t_disp = _get_par(dc_tdwell, dispstat)
anchor, a_disp = _get_par(dc_anchor, dispstat)
anchor *= np.pi/2
print("basis:")
print('tdwell', tdwell)
print('anchor', anchor)
# %%
# should compute constraction from piuls.n_retractions ?
a_disp

# %%
# get target  value
_tickfsize = 18
_fsize = 24
_mplcf = {}
_mplcf["legend.fontsize"] = 18

m0 = {'c':'indianred', 'marker':'o', 'alpha':1.0}
m1 = {'c':'salmon', 'marker':'v', 'alpha':1.0, 'linestyle':'--'}
mscat = {'edgecolor':'white', 'linewidth' : 1.0, 'alpha' : 0.5, 's': 60}
target_contraction = -1 * twutils.make_get('effective_contract_length.mean')(ltarget)
target_disp = twutils.make_get('pdisp.median')(ltarget)
print('target median displacement = {}'.format(target_disp))
targetstyle = {'s':160, 'c':'r', 'marker':'x', 'linewidth':4}
# plot relationships dwell_time v delta_l and \alpha v  delta_lj
params = sim4d["params"]
params["contraction.mean"] = -1 * sim4d["data"].get("effective_contract_length.mean")
params["contraction.median"] = -1 * sim4d["data"].get("effective_contract_length.median")
acc_params = params.iloc[_accepted.index]
#
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(acc_params['dwell_time'], acc_params['contraction.mean'], label="accepted", **mscat)
ax.scatter(simpar['dwell_time'], target_contraction, **targetstyle)
ax.plot(tdwell, -1*tdwell_contraction, label=r'$\langle \Delta l \rangle$', **m0)
ax.plot(tdwell, t_disp, label=r'$\langle \Delta x \rangle$', **m1)
ax.set_ylabel(r'$\mu m$', fontsize=_fsize)
ax.set_xlabel(r'$\tau_{\mathrm{dwell}}$', fontsize=_fsize)
ax.legend(fontsize=_mplcf["legend.fontsize"])
ax.tick_params(axis='both', which='major', labelsize=_tickfsize)
ax.grid(False)
fig.tight_layout()
pub.save_figure('tdwell_v_contraction', notename, fig=fig)

#
fig, ax = plt.subplots(figsize=(5,5))
_acc = ax.scatter(
	acc_params['anchor_angle_smoothing_fraction'], acc_params['contraction.mean'], 
	label='accepted', **mscat)
_tar = ax.scatter(
	simpar['anchor_angle_smoothing_fraction'], target_contraction, 
	**targetstyle)
_slt = ax.plot(anchor, -1*anchor_contraction, label=r'$\langle \Delta l \rangle$', **m0)
_ldisp = ax.plot(anchor, a_disp, label=r'$\langle \Delta x \rangle$', **m1)
_leg= ax.legend(fontsize=_mplcf["legend.fontsize"])
# ax.set_ylabel(r'$\langle \Delta l \rangle$')
ax.set_ylabel(r'$\mu m$', fontsize=30)
ax.set_xlabel(r'$\alpha$', fontsize=30)
# ax.axhline(0.12, c='k', linestyle='--')
# _leg= ax.legend([_acc, _tar, _slt], ['accepted', 'reference', 'sim'])
ax.tick_params(axis='both', which='major', labelsize=_tickfsize)
ax.grid(False)
fig.tight_layout()
pub.save_figure('anchor_v_contraction', notename, fig=fig)

# %%
# what is the relationship between contraction.mean and activity
fig, ax = plt.subplots(figsize=(5,5))
# sns.scatterplot(data=params, x='contraction.mean', y='ahat.estimate', ax=ax)
sns.scatterplot(data=acc_params, x='contraction.mean', y='ahat.estimate')
ax.scatter(target_contraction, simref['ahat.estimate'], **targetstyle)

# %%
# -- what about anchor parameter and activity?
# 
_anchor, a_ahat = _get_par(dc_anchor, 'ahat.estimate')
fig, ax  = plt.subplots(figsize=(5,5))
ax.scatter(acc_params['anchor_angle_smoothing_fraction'], acc_params["ahat.estimate"], 
	label="accepted", **mscat)
ax.plot(anchor, a_ahat, label="sim", **m0)
ax.scatter(simpar["anchor_angle_smoothing_fraction"], simref["ahat.estimate"], **targetstyle)
ax.set_ylabel(r'$\hat{a}$', fontsize=_fsize)
ax.set_xlabel(r'$\alpha$', fontsize=_fsize)
ax.set_ylim((0,0.3))
ax.legend()
pub.save_figure('anchor_v_ahat', notename, fig=fig)

_tdwell, t_ahat = _get_par(dc_tdwell, 'ahat.estimate')
fig, ax  = plt.subplots(figsize=(5,5))
ax.scatter(acc_params['dwell_time'], acc_params["ahat.estimate"], 
	label="accepted", **mscat)
ax.plot(tdwell, t_ahat, label="sim", **m0)
ax.scatter(simpar["dwell_time"], simref["ahat.estimate"], **targetstyle)
ax.set_ylabel(r'$\hat{a}$', fontsize=_fsize)
ax.set_xlabel(r'$\tau_{\mathrm{dwell}}$', fontsize=_fsize)
# ax.set_ylim((0,0.3))
ax.legend()



# %%
# so despite having k_spawn ~ 1 and alpha small, we can achieve mean of at least 0.02 microns/s
# lets look at this in more detail
unt_accepted.sort_values('k_spawn')


# %%
# all pairs
if plotting:
	import itertools
	_objectives = ["lvel.mean", "deviation.var", "ahat.estimate"]
	for pair in itertools.combinations(_objectives, 2):
		_regdf = statdf[sim4d["problem"]["names"] + list(pair)]
		_accepted = abcimplement.rejection_abc(_regdf, list(pair), statref, N)
		fig, axes = abcimplement.perfectplot4d(sim4d["problem"], _accepted, simpar=simpar)
		fig.suptitle(pair)
		plt.tight_layout()


# %%
# recompute the accepted samples based on lvel alone
lvel_accepted = abcimplement.rejection_abc(statdf, ['lvel.mean'], statref, N)
lpar = [r'$\tau_{\mathrm{dwell}}$', r'$\kappa$', r'$\alpha$', r'$k_{\mathrm{spawn}}$']
p1, p2 = ["anchor_angle_smoothing_fraction", "k_spawn"]
xdata, ydata = lvel_accepted[p1], lvel_accepted[p2]
# scipy curve fitting
# candidate function
# https://math.stackexchange.com/questions/2355818/a-curve-with-two-specific-asymptotic-lines/2355836
def candf(x, a, b):
	# return np.exp(-b*x)/(a*x) 
	return np.exp(-b*x)/(a*x) 
	# return 1/(a*x)
p, pcov = scipy.optimize.curve_fit(candf, xdata, ydata)
print('a,b', p)
basis = np.linspace(0.01,1.5,1500)
fit = [candf(x, *p) for x in basis]

# %%
fig, ax = plt.subplots(figsize=(5,5))
# ax.plot(basis, fit, label='fit', color='k', linestyle='--')
ax.scatter(xdata, ydata, **mscat)
ax.set_xlabel(lpar[2], fontsize=_fsize)
ax.set_ylabel(lpar[3], fontsize=_fsize)
ax.tick_params(axis='both', which='major', labelsize=_tickfsize)
ax.grid(False)
ax.set_xlim(0, np.max(xdata))
ax.set_ylim(0, np.max(ydata))
# fit a curve to this?
fig.tight_layout()
pub.save_figure("lvel_curve", notename, fig=fig)


# %%

# retreive the data for the optimal accepted sample
udir = join(sim4d["simdir"], sim4d["lookup"][0][prime_accepted.index[0]])
# twutils.sync_directory(udir)
_trs = readtrack.trackset(ddir=join(udir, 'data/'))
bestsim = [_fj.linearize(tr) for tr in _trs]
prime_accepted.iloc[0]

# %%
# best sample, velocity distribution comparison

with plt.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(6,5))
	sim_vel = np.concatenate([ltr.get_step_speed() for ltr in bestsim])
	xlim = (0, np.quantile(sim_vel, 0.98))
	ref_vel = _fj.load_subset_speed()["top"]
	ks, p = scipy.stats.ks_2samp(ref_vel, sim_vel)
	print("ks statistic = {:0.3f}".format(ks))
	shstyle = dict(element="step", fill=True, alpha=0.3)
	common = {"bins": 50, 'binrange':xlim, **shstyle}
	sns.histplot(ref_vel, ax=ax, stat="density", label="reference", **common)
	sns.histplot(sim_vel, ax=ax, stat="density", color="orangered", label="simulation", **common)
	# sns.histplot(t3_vel, ax=ax, stat="density", color="orangered", alpha=0.5, label="sim", **common)
	ax.set_xlim(xlim)
	ax.legend(fontsize=28)
	ax.set_xlabel(r"step velocity (\textmu m/s)", fontsize=30)
	ax.set_ylabel("density", fontsize=30)
	ax.grid(False)   
plt.tight_layout()
pub.save_figure("_best_sim_crawling_lvel_distrib", notename)


# %%
# slice the original track and then linearize
_extr = readtrack.TrackLike(_trs[0]._track[:2000])
extr = _fj.linearize(_extr)
import plotutils


texstyle = {"font.size": 22, "text.usetex": True, "axes.labelsize" : 26}
with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=(10,2))
	plotutils.plot_lvel(ax, extr)
	# looks similar
	ax.set_ylim((0,1))
	_fsize=26
	ax.set_ylabel(r"v\,(\textmu m/s)", fontsize=_fsize)
	# ax.set_xlabel(r"time (s)", fontsize=_fsize)
	ax.set_xticks([])
	_score = _accepted['score'].iloc[0]
	print('best score', _score)
	# ax.text(0.40, 0.8, "$\rho$ = {:.4f}".format(_score), transform=ax.transAxes, fontsize=26)
	ax.grid(False)
	ax.yaxis.set_ticks([0,0.5,1.0])
	# ax.yaxis.set_major_locator(plt.MaxNLocator(4))
pub.save_figure("sim_top_best_crawling", notename, fig=fig)

pub.save_thesis_figure('sim_top_best_crawling_vel_profile')

# %%
subsetdata = _fj.load_subsets()
subsetidx = _fj.load_subset_idx()

# %%
# 
i = 12
i = 39
i = 30
i = 26

top = subsetdata["top"]
track_idx = subsetidx["top"][i]
print('track', track_idx)

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=(10, 2.0))
	ax.set_ylim((0,1))
	l = plotutils.plot_lvel(ax, top[i], xcut=200)
	ax.set_ylabel(r"v\,(\textmu m/s)", fontsize=_fsize)
	ax.set_xlabel(r"time (s)", fontsize=_fsize)
	# track 1215
	# ax.text(0.40, 0.8, "track id = {}".format(track_idx), transform=ax.transAxes, fontsize=26)
	ax.grid(False)
	ax.yaxis.set_ticks([0,0.5,1.0])
pub.save_figure("fj_example_velocity_profile", notename)

pub.save_thesis_figure('fj_example_velocity_profile')

# %%

_trs[0]
'best sim parameters', sim4d['lookup'][1][udir.split('/')[-1]]
best_idx = (sim4d['params']['uid'] == udir.split('/')[-1]).to_list().index(True)
sim4d['params'].iloc[best_idx]

# %%
# plot the trajectory

import shapeplot

_extr = readtrack.TrackLike(_trs[0]._track[:1001])

fig, ax = plt.subplots(figsize=(12,4))
shapeplot.sim_capsdraw(ax, _extr, 100, style={'alpha':0.3, 'lw':6})
# .longtracks(ax, _extr)
lstyle = dict(lw=6, alpha=0.8)
h1, = ax.plot(_extr['x'], _extr['y'], **lstyle)
h2, = ax.plot(_extr['trail_x'], _extr['trail_y'], **lstyle)
ax.axis(False)

# pt = np.array([0,0])
# ax.plot([pt[0],pt[0]+1.0], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)
# delta = 0.005
# ax.text(pt[0]+0.1 + delta + 0.005, pt[1]-delta-0.001, "$0.1\mu m$", fontsize=20)

ax.legend([h1, h2], ["leading", "trailing"], fontsize=24)

pub.save_thesis_figure('example_simulated_shapeplot')

# %%
# -5, -41, -53
_extr = top[-54]
_extr['time'] = _extr["time"] - _extr['time'][0]
_extr = readtrack.TrackLike(_extr._track[:1001])
def flip(tr):
	tr = tr.copy()
	tr['x'] = -tr['x']
	tr['y'] = -tr['y']
	tr['trail_x'] = -tr['trail_x']
	tr['trail_y'] = -tr['trail_y']
	return tr
_extr = flip(_extr)

fig, ax = plt.subplots(figsize=(12,4))
shapeplot.sim_capsdraw(ax, _extr, 100, style={'alpha':0.3, 'lw':7})
# .longtracks(ax, _extr)
_lstyle = lstyle.copy()
_lstyle["lw"] = 6
ax.plot(_extr['x'], _extr['y'], **_lstyle)
ax.plot(_extr['trail_x'], _extr['trail_y'], **_lstyle)
ax.axis(False)


pub.save_thesis_figure('example_simulated_fanjin_track')
#

# %%

plt.figure()
vthreshold = 0.25
tr = top[i]
speed = tr.get_step_speed()
fastidx = speed > vthreshold
fastidx
x = tr['x'][tr.step_idx]
y = tr['y'][tr.step_idx]
plt.plot(x, y, marker='o')
fastx = x[:-1][fastidx]
fasty = y[:-1][fastidx]
idx = np.argwhere(fastidx)
idx

for _i in idx:
	_x = [x[_i], x[_i+1]]
	_y = [y[_i], y[_i+1]]
	plt.plot(_x, _y, c='r')


# %%
# compute min/max for accepted samples for each parameter
print("quantiles of accepted samples!")
prime_accepted
qmin, qmax = 0.00,  1.00
qmin, qmax = 0.05,  0.95
print("quantiles = ", (qmin,  qmax))
qr = {}
for par, value in simpar.items():
	parr = prime_accepted[par]
	pair = np.quantile(parr, qmin), np.quantile(parr, qmax)
	qr[par] = pair
qr

# %%
bound_pili_participation = sim4d["data"].get("bound_pili_participation")
print("mean bound_pili_participation", np.mean(bound_pili_participation))
print("mean accepted bound_pili_participation", np.mean(bound_pili_participation[prime_accepted.index]))


# %%
# relationships between summary statistics in accepted samples?
# fig, ax = plt.subplots(figsize=(5,5))
# sns.scatterplot(data=unt_accepted, x="deviation.var", y="ahat.estimate", ax=ax)


# %%
