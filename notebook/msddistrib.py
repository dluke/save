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
# 
# analyse velocity distributions / MSD / autocorrelation
# we have done this many times before but perhaps we can be more thorough

# relevant papers
# Persistent Cell Motion in the Absence of External Signals: A Search Strategy for Eukaryotic Cells
# https://www.researchgate.net/publication/5293896_Persistent_Cell_Motion_in_the_Absence_of_External_Signals_A_Search_Strategy_for_Eukaryotic_Cellshttps://www.researchgate.net/publication/5293896_Persistent_Cell_Motion_in_the_Absence_of_External_Signals_A_Search_Strategy_for_Eukaryotic_Cells

# https://www.sciencedirect.com/science/article/pii/S0378437120302168

# we also find anticorrelation in the angles of adjacent steps similar to dicty

# %% 
import sys, os
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
norm = np.linalg.norm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import pili
from pili import support
from support import get_array, make_get
import _fj
import scipy.stats
import twanalyse
import sobol
import pwlstats
import pwlpartition

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili.publication as publication
print("writing figures to", publication.writedir)

import thesis.publication as thesis

verbose = False
notename = "msddistrib"


# %% 
# start by analysing the candidate trajectory
# * original? smoothed?
track_idx = 2924
candidate = _fj.trackload_original([track_idx])[0]
fjwavetrack = _fj.trackload([track_idx])[0]
lintrack = _fj.lintrackload([track_idx])[0]

# %% 

track = candidate


# scaling = np.linspace(1,50)
geomscaling = np.array([1,2,3,5,8,13,20,32,50,100,200,300,500])

def compute_msd(track, scaling=geomscaling):
	xy = track.get_head2d()
	x_, y_ = xy[0,0], xy[0,1]
	x, y = xy[:,0] - x_, xy[:,1] - y_ # subtract the starting position
	msd_n = []
	for i, window in enumerate(scaling):
		msd = (x[window:] - x[:-window])**2 +  (y[window:] - y[:-window])**2 
		msd_n.append(msd)
	return msd_n

msd_n = compute_msd(track, scaling=geomscaling)

# %% 

for i, msd in enumerate(msd_n):
	fig, ax = plt.subplots(figsize=(6,4))
	sns.histplot(msd)
	ax.set_title(r"$\tau = {:0.1f}$".format(geomscaling[i]*0.1))

# \tau < 2 : Exponential decay (?)
# \tau > 2 : peak centralises (gaussian like)

# %% 

scaling = geomscaling

# compute the MSD
meansd = [np.mean(msd) for msd in msd_n]
with mpl.rc_context({'font.size':20}):
	fig, ax = plt.subplots(figsize=(6,4))
	x = scaling*0.1
	ax.plot(scaling*0.1, meansd/x, marker='x')

	ax.set_xlabel(r"$\tau$")
	ax.set_ylabel(r"MSD")

# %% 
# TODO compute these curves for all trajectories
# !load data
def load_classification(path):
	load_path = join(pili.root, path)
	idx_list = np.loadtxt(load_path).astype(int)
	return _fj.trackload_original(idx_list)
crawling_track_list = load_classification('notebook/thesis_classification/crawling.npy')
walking_track_list = load_classification('notebook/thesis_classification/walking.npy')

#! load top subset
top_track_list = _fj.trackload_original(_fj.slicehelper.load('candidates_whitelist'))

# %% 
# !compute curves

_geomscaling = np.array([1,2,3,5,8,13,20,32,50,100,200,300,500])

def compute_msd_curve_list(track_list):
	msd_curve_list = []
	for track in track_list:
		_msd = compute_msd(track, scaling=_geomscaling)
		msd_curve_list.append([np.mean(_m) for _m in _msd])
	return msd_curve_list

crawling_msd_curve = compute_msd_curve_list(crawling_track_list)
walking_msd_curve = compute_msd_curve_list(walking_track_list)
top_msd_curve = compute_msd_curve_list(top_track_list)


# %% 
# !plotting

c_color = '#065DBC'
w_color = '#BC062A' 

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(6,4))
	xspace = 0.1 * _geomscaling

	# guides
	guidestyle = dict(c='k', linestyle='--', alpha=0.8)
	guide_1 = (xspace, 0.003*xspace) 
	guide_2 = xspace, 0.03*(1.5*xspace)**2
	ax.plot(*guide_1, **guidestyle)
	ax.plot(*guide_2, **guidestyle)

	msdstyle = dict(alpha=0.6, lw=2.5, markersize=8)
	# for curve in crawling_msd_curve[::100]:
	# 	h1, = ax.loglog(xspace, curve, color=c_color, **msdstyle)

	# use high speed crawling tracks
	for curve in top_msd_curve[::18]:
		h1, = ax.loglog(xspace, curve, color=c_color, marker='x', **msdstyle)
	print(len(top_msd_curve[::18]))

	for curve in walking_msd_curve[4::30]:
		h2, = ax.loglog(xspace, curve, color=w_color, marker='o', fillstyle='none', **msdstyle)
	print(len(walking_msd_curve[4::30]))

	ax.set_xlabel(r'timescale $\tau$ (s)')
	ax.set_ylabel(r'$\langle \delta(\tau)^2 \rangle$')


	rectstyle = dict(alpha=0.2, color='orange')
	xy = (guide_1[0][0], guide_1[1][0])
	patch = mpl.patches.Rectangle(xy, width = 0.9, height = 1, **rectstyle)
	ax.add_patch(patch)
	ax.set_xticks([0.1,1.0,10,50], labels=[0.1,1.0,10,50])

	handles = [h1, h2]
	labels = ['crawling ', 'walking']

	ax.legend(handles, labels, loc='upper left')

	xy = np.array([0.7, 0.25])
	ax.annotate(r'$k = 1$', xy, textcoords = 'axes fraction')
	xy = np.array([0.6, 0.82])
	ax.annotate(r'$k = 2$', xy, textcoords = 'axes fraction')
	# 	# ax.annotate(r'$\delta_{\mathrm{step}}$', (0.12, 10), xytext=(0.6, 0.5), textcoords='axes fraction', fontsize=16, arrowprops=dict(facecolor='black', width=1, shrink=0.05, headwidth=4))

publication.save_figure('msd_curves')

# %% 
#! now draw the distributions of kMSD as a complimentary figure

def fit_kmsd(basis, msd, weights=None):
	basis = np.array(basis)
	msd = np.array(msd)
	select = basis > 10
	l_basis, l_msd = np.log(basis[select]), np.log(msd[select])
	p, cov = np.polyfit(l_basis, l_msd, deg=1, cov=True)
	return p[0]


with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(6,4))
	top_kmsd = [fit_kmsd(_geomscaling, _msd) for _msd in top_msd_curve]
	walking_kmsd = [fit_kmsd(_geomscaling, _msd) for _msd in walking_msd_curve]
	hstyle = dict(alpha=0.7)
	sns.histplot(top_kmsd, binrange=(0,2.0), bins=20, color=c_color, label='crawling', **hstyle)
	sns.histplot(walking_kmsd, binrange=(0,2.0), bins=20, color=w_color, label='walking', **hstyle)
	ax.xaxis.set_major_locator(plt.MaxNLocator(5))
	# ax.set_xlabel(r"$k_{\textrm{MSD}}$")
	ax.set_xlabel(r"$k$")
	ax.legend(loc=(0.01, 0.6), handlelength=1.0)
	# ax.legend([h1, h2], ["crawling", "walking"])

	ax.yaxis.set_major_locator(plt.MaxNLocator(5))


publication.save_figure('kmsd_histplot')


# %% 
# ! the same figure but use plos paper walking subset

# !EXP
walking = _fj.load_subsets()["walking"]
walking_msd_curve = compute_msd_curve_list(walking)

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(6,4))
	top_kmsd = [fit_kmsd(_geomscaling, _msd) for _msd in top_msd_curve]
	walking_kmsd = [fit_kmsd(_geomscaling, _msd) for _msd in walking_msd_curve]
	hstyle = dict(alpha=0.7)
	sns.histplot(walking_kmsd, binrange=(0,2.0), bins=20, color=w_color, label='walking', **hstyle)
	sns.histplot(top_kmsd, binrange=(0,2.0), bins=20, color=c_color, label='crawling', **hstyle)
	ax.xaxis.set_major_locator(plt.MaxNLocator(5))
	# ax.set_xlabel(r"$k_{\textrm{MSD}}$")
	ax.set_xlabel(r"$k$")
	ax.legend(loc=(0.01, 0.6), handlelength=1.0)
	# ax.legend([h1, h2], ["crawling", "walking"])

	ax.yaxis.set_major_locator(plt.MaxNLocator(5))


publication.save_figure('plos_kmsd_histplot')





# %% 

fig, ax = plt.subplots(figsize=(6,4))
def make_dicty(tp, vp):
	def dicty(tau):
		vp2 = vp**2
		return 2*tp*vp2 * (tau - tp * (1 - np.exp(-tau/tp)))
		# return 2*tp*vp2 * tau
	return dicty

tp = 8.8 
vp = 5.4 
dicty = make_dicty(tp, vp)

_scale = np.linspace(0.1,1000,10000)

# ax.loglog(_scale, list(map(dicty, _scale)))
# ax.plot(_scale, list(map(dicty, _scale)))
ax.plot(_scale, np.array(list(map(dicty, _scale)))/_scale)



# %% 
# obtain the body parallel and perpendicular components of velocity
# velocity will be defined using \tau
# ! switch to FJ linearised data

scale_dstep = np.array([0.03,0.06,0.12,0.24,0.36,0.48,0.60,0.80,1.0,1.2,1.5,2.0,5.0])
lindata = [_fj.linearize(fjwavetrack, step_d=dstep) for dstep in scale_dstep]

def body_velocity(track):
	vel = track.get_step_velocity()
	body = track.get_body_projection()[track.step_idx[:-1]]
	bodyaxis  = body/norm(body,axis=1)[:,np.newaxis]
	bodyperp = np.empty_like(bodyaxis)
	bodyperp[:,[0, 1]] = bodyaxis[:,[1, 0]]

	vll = (bodyaxis * vel).sum(axis=1)
	vlt = (bodyperp * vel).sum(axis=1)
	return vll, vlt

for i, track in enumerate(lindata):
	vll, vlt = body_velocity(track) 

	
	with mpl.rc_context({'font.size':16}):
		fig, axes  = plt.subplots(1, 2, figsize=(8,4))
		ax = axes[0]
		sns.histplot(vll,ax=ax)
		ax = axes[1]
		sns.histplot(vlt,ax=ax)
		fig.suptitle(r'$\delta = {:.2f}$'.format(scale_dstep[i]))

# * vary the linearisation distance?
# like/unlike dicty?

# %% 
# * similar analysis but for PWL model, look for alternating perpendicular displacements here too
path = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
pwl_model = pwlstats.load_model_at(path)

len(pwl_model)

# %% 
# * pwl model doesn't know about the body, see pwlstats.empirical_summary

model = pwl_model
def get_deviation(model):
	segmentv = model.get_vectors()
	body = track.get_body_projection()
	time = model.get_time()
	return pwlstats.angle(body[time[:-1]], segmentv)
deviation  = get_deviation(model)

fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(deviation, bins=30, ax=ax)
# * need to adjust bins but there is a slight dip at the center
# * the next question is whether displacements alternate

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(deviation[:-1], deviation[1:])
lstyle = dict(color='k', alpha=0.4)
ax.axvline(0, **lstyle)
ax.axhline(0, **lstyle)
ax.set_aspect("equal")

# anticorrelation in turn directions is visible
coef, pvalue  = scipy.stats.pearsonr(deviation[:-1], deviation[1:])
"correlation", coef, pvalue  

# * pvalue = 0.08, significant enough?
# * we only have ~150 samples

# %% 
# * the same but use the lab frame? ~  the body frame is changing slightly
def plot_alpha_distrib(ax, alpha):
	sns.histplot(alpha, bins=30, ax=ax)
	ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2, np.pi])
	ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"])
	ax.set_xlabel(r"$\theta$")

def plot_alpha_scatter(ax, alpha, mstyle={}):
	mscat = {'edgecolor':'white', 'linewidth' : 1.0, 'alpha' : 1.0, 's': 60}
	mscat.update(mstyle)
	ax.scatter(alpha[:-1], alpha[1:], **mscat)
	lstyle = dict(color='k', alpha=0.4)
	ax.axvline(0, **lstyle)
	ax.axhline(0, **lstyle)
	ax.set_aspect("equal")

	buff = 1.1
	lims = (-buff*np.pi, buff*np.pi)
	ax.set_xlim(lims)
	ax.set_ylim(lims)
	ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2, np.pi])
	ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"])
	ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2, np.pi])
	ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"])

	ax.set_xlabel(r"$\theta_j$")
	ax.set_ylabel(r"$\theta_{j+1}$")

v  = model.get_vectors()
alpha = pwlstats.angle(v[:-1], v[1:])

with mpl.rc_context({'font.size':20}):
	fig, axes = plt.subplots(1,2,figsize=(10,5))
	ax = axes[0]
	plot_alpha_distrib(ax, alpha)

	ax = axes[1]
	plot_alpha_scatter(ax, alpha)
	plt.tight_layout()

pub.save_figure("lab_angle_correlation_pwl_model", notename)


coef, pvalue  = scipy.stats.pearsonr(alpha[:-1], alpha[1:])
"correlation", coef, pvalue  

# sparseml/run/cluster/no_heuristic/top

# %% 
# * Is this feature visible in wavelet steps?
stepdx = lintrack.get_step_dx()
step_angle = pwlstats.angle(stepdx[:-1], stepdx[1:])

with mpl.rc_context({'font.size':20}):
	fig, axes = plt.subplots(1,2,figsize=(10,5))
	ax = axes[0]
	plot_alpha_distrib(ax, step_angle)

	ax = axes[1]
	plot_alpha_scatter(ax, step_angle)
	fig.suptitle("dstep = {:.2f}".format(0.12))
	plt.tight_layout()

pub.save_figure("lab_angle_correlation_dstep=0.12", notename)


# %% 
if verbose:
	# * not even if we vary the dstep?
	for i, track in enumerate(lindata):
		stepdx = track.get_step_dx()
		step_angle = pwlstats.angle(stepdx[:-1], stepdx[1:])
		with mpl.rc_context({'font.size':20}):
			fig, axes = plt.subplots(1,2,figsize=(10,5))
			fig.suptitle(r'$\delta = {:.2f}$'.format(scale_dstep[i]))
			ax = axes[0]
			plot_alpha_distrib(ax, step_angle)

			ax = axes[1]
			plot_alpha_scatter(ax, step_angle)
			plt.tight_layout()

			coef, pvalue  = scipy.stats.pearsonr(step_angle[:-1], step_angle[1:])
			print("correlation at dstep = {:.2f}".format(scale_dstep[i]), coef, pvalue)

# according to pvalue the anticorrelation is most visible at 0.36 - 0.60 linearised steps

# plot the data for dstep = 0.36
i = scale_dstep.tolist().index(0.36)

track = lindata[i]
stepdx = track.get_step_dx()
step_angle = pwlstats.angle(stepdx[:-1], stepdx[1:])
with mpl.rc_context({'font.size':20}):
	fig, axes = plt.subplots(1,2,figsize=(10,5))
	fig.suptitle(r'dstep = {:.2f}'.format(scale_dstep[i]))
	ax = axes[0]
	plot_alpha_distrib(ax, step_angle)

	ax = axes[1]
	plot_alpha_scatter(ax, step_angle)
	plt.tight_layout()

	coef, pvalue  = scipy.stats.pearsonr(step_angle[:-1], step_angle[1:])
	print('N = ', track.get_nsteps())
	print("correlation at dstep = {:.2f}".format(scale_dstep[i]), coef, pvalue)

pub.save_figure("lab_angle_correlation_dstep=0.36", notename)

# %% 
# * what longer time/distance correlations?
# todo

# %% 
# * the same but use the candidate_whitelist
select_idx = _fj.load_subset_idx()["top"]
toptrs = _fj.trackload(select_idx)
print("loaded {} tracks".format(len(toptrs)))

dstep = 0.36
lintoptrs = [_fj.linearize(tr, step_d=dstep) for tr in toptrs]


# %% 
dxdata = [tr.get_step_dx() for tr in lintoptrs]
dxu = np.concatenate([dx[:-1] for dx in dxdata])
dxv = np.concatenate([dx[1:] for dx in dxdata])
alpha = pwlstats.angle(dxu, dxv)

with mpl.rc_context({'font.size':20}):
	fig, axes = plt.subplots(1,2,figsize=(10,5))
	fig.suptitle(r'$\delta = {:.2f}$'.format(dstep))
	ax = axes[0]
	plot_alpha_distrib(ax, alpha)

	ax = axes[1]
	mstyle = {'s':20}
	plot_alpha_scatter(ax, alpha, mstyle)
	plt.tight_layout()

	coef, pvalue  = scipy.stats.pearsonr(alpha[:-1], alpha[1:])
	print('N = ', len(alpha))
	print("correlation at dstep = {:.2f}".format(scale_dstep[i]), coef, pvalue)

# pub.save_figure("top_lab_angle_correlation_dstep=0.36", notename)


# %% 
# * the same but take a chunk of pwl solved trajectories
# sparseml/run/cluster/no_heuristic/top
from glob import glob
pwlpartition.hide_output = True

# found = sorted(glob(join(pwlstats.root, "run/cluster/no_heuristic/top/_top_*")))
found = [join(pwlstats.root, f"run/cluster/no_heuristic/top/_top_{idx:04d}")  for idx in select_idx]
pwllist= [pwlstats.load_model_at(directory) for directory in found]
pwlpair = [(k,v) for k, v in zip(select_idx, pwllist) if v != None]
haveidx, pwldata = zip(*pwlpair)

dxdata = [model.get_vectors() for model in pwldata]
dxu = np.concatenate([dx[:-1] for dx in dxdata])
dxv = np.concatenate([dx[1:] for dx in dxdata])
alpha = pwlstats.angle(dxu, dxv)

with mpl.rc_context({'font.size':20}):
	fig, axes = plt.subplots(1,2,figsize=(10,5))
	ax = axes[0]
	plot_alpha_distrib(ax, alpha)

	ax = axes[1]
	mstyle = {'s':20}
	plot_alpha_scatter(ax, alpha, mstyle)
	plt.tight_layout()

	coef, pvalue  = scipy.stats.pearsonr(alpha[:-1], alpha[1:])
	print('N = ', len(alpha))

	print("pwl angle correlation", coef, pvalue)


# solver = pwlpartition.Solver.load_state(join(pwlstats.root, f"run/cluster/no_heuristic/top/_top_{track_idx:04d}/", "solver"))

# %% 
# * same thing seen in simulation?
# start with the crawling target
simtarget = "/home/dan/usb_twitching/run/825bd8f/target/t0"

simpwl, _ = pwlstats.load_sim_pwl(simtarget)

dxdata = [model.get_vectors() for model in simpwl]

print('count nans', [np.sum(norm(dxdata[i],axis=1) == 0.) for i in range(len(dxdata))])
for i, dx in enumerate(dxdata):
	dx = dx[~(norm(dx,axis=1)==0.)] 
	dxdata[i] = dx

# %% 
dxu = np.concatenate([dx[:-1] for dx in dxdata])
dxv = np.concatenate([dx[1:] for dx in dxdata])
alpha = pwlstats.angle(dxu, dxv)

with mpl.rc_context({'font.size':20}):
	fig, axes = plt.subplots(1,2,figsize=(10,5))
	ax = axes[0]
	plot_alpha_distrib(ax, alpha)

	ax = axes[1]
	mstyle = {'s':20}
	plot_alpha_scatter(ax, alpha, mstyle)
	plt.tight_layout()

	coef, pvalue  = scipy.stats.pearsonr(alpha[:-1], alpha[1:])
	print('N = ', len(alpha))

	print("pwl angle correlation", coef, pvalue)

pub.save_figure("simt0_lab_correlation_scatter", notename)

# %% 
# * add pwl angle correlation statistic to summary
simdir = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d/"
def load_abc_pwl(simdir):
	summary = lambda uid : pwlstats.sim_summary(join(simdir, uid))
	lookup = sobol.read_lookup(simdir)
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

with support.Timer():
	valid, ldata, tdata = load_abc_pwl(simdir)

# %% 
# valid
corrls = [local["angle"]["corrcoef"][0] for local in ldata]
# need to construct the dataframe
sns.histplot(corrls)

# %% 
simdir = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d/"
lookup = sobol.read_lookup(simdir)
problem = sobol.read_problem(simdir)
lduid = sobol.collect_lduid(simdir)
problem
# %% 

def compile_params(problem, lookup, valid):
	_parlist = list(zip(problem["names"],  zip(*[lookup[1][_u] for _u in valid])))
	_col = {}
	_col["uid"] = [_u  for _u in valid]
	_col.update({k:v for k, v in _parlist})
	return pd.DataFrame(_col)
params = compile_params(problem, lookup, valid)
params["angle_corr"] = corrls

lduidlist = [lduid[_u] for _u in valid]
objectives = ["lvel.mean", "nbound.mean", "ntaut.mean"]
for obs in objectives:
	var = get_array(make_get(obs), lduidlist)
	params[obs] = var
params


# %% 

fix_dwell = params[params["dwell_time"].between(0.8,1.2)]
fix_alpha = fix_dwell[fix_dwell["anchor_angle_smoothing_fraction"].between(0.2,0.5)]
fix_pilivar = fix_alpha[fix_alpha["pilivar"].between(2.0,3.0)]
fix = fix_pilivar
fix = fix_alpha 
print('sliced {} samples'.format(len(fix)))

sns.scatterplot(data=fix, x="nbound.mean", y="angle_corr")


