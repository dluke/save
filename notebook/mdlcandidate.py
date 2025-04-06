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
# obtain the estimated distributions for a human verified, solved, experimental trajectory

# %% 
import os
import json
import pickle
import numpy as np
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd
from glob import glob
import scipy.stats

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import thesis.publication as thesis

import pili
import mdl
import annealing 

import pwlstats
import pwlpartition

import _fj

# %% 
notename = "mdlcandidate"
mplstyle = {"font.size": 20}

# %% 

def load_json(at):
	with open(at, 'r') as f: return json.load(f)

target = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")

config = load_json(join(target, 'config.json'))
track = _fj.trackload_original([config["trackidx"]])[0]


# %%

solver = pwlpartition.Solver.load_state(join(target, "solver"))
# !tmp
solver.partition.use_probability_loss = False
solver.partition.inter_term = False
# !~tmp

candidate_summary = pwlstats.solver_summary(solver)
candidate_sigma = candidate_summary["estimate_sigma"]
candidate_r = pwlpartition.estimate_r(candidate_sigma)
draw_sigma_l = candidate_sigma / candidate_summary["mean_step_length"]
draw_rl = candidate_r / candidate_summary["mean_step_length"]
# todo use 0.05 quantile instead of mean ~ we want to find all the segments
print("esimate experimental sigma/l {:.4f}/{:.4f} = {}".format(candidate_sigma , candidate_summary["mean_step_length"], draw_sigma_l))
print("esimate experimental r/l", draw_rl)
candidate_summary

# %% [markdown]
# want the empirical distribution 
# * for segment lengths
# * segment angle relative the body axis
# * segment angle relative to the last segment
# * sampling distribution ~ denoise curve coordinate?

# %% 
# what is the error in the body axis?
from skimage.restoration import estimate_sigma, denoise_wavelet

def plot_body_axis(track):
	N = 200
	body = track.get_body_projection()
	lab_theta = np.arctan2(body[:,1], body[:,0])

	fig, ax = plt.subplots(figsize=(10,4))

	model = solver.partition.model
	time = model.get_time()
	max_index = np.searchsorted(time, N)
	for index in time[:max_index]:
		ax.axvline(index, linestyle='--', color='black', alpha=0.4)
	ax.plot(lab_theta[:N], label='body angle')
	theta_sigma = estimate_sigma(lab_theta)
	# print('estimate error in body axis', estimate_sigma(lab_theta))
	# ax.plot(denoise_wavelet(lab_theta[:N], sigma=theta_sigma), label='denoised')
	ax.legend()

plot_body_axis(track)


# %% 
# what is the error in the curve coordinate?
# in mdlvelocity.py we also analyse the curve coordinate
def plot_curve_coord(solver):
	fig, ax = plt.subplots(figsize=(20,8))

	curve_coord = solver.partition.get_curve_coord()
	N = 50
	ax.plot(curve_coord[:N], alpha=0.6)

	curve_coord_sigma = estimate_sigma(curve_coord)
	print('estimate sigma = ', curve_coord_sigma)

	wave_config = {"wavelet":'db1', 'method':'BayesShrink', "mode":'soft'}
	_sigma = 2*curve_coord_sigma
	wave_curve_coord = denoise_wavelet(curve_coord, sigma=_sigma, **wave_config)
	ax.plot(wave_curve_coord[:N], alpha=0.6)

plot_curve_coord(solver)

# %% 
from pwlstats import fit_exp_function, fit_exp_curve, normal_function, fit_normal_curve

local = pwlstats.empirical_summary(solver, track)

def describe(data):
	print("N = ", data.size)
	print("mean", np.mean(data))
	print("median", np.median(data))
	
# %% 
# outliers in dx could mean that the curve is fitted poorly
def plot_outliers():
	model = solver.partition.model
	dx = local["dx"]
	time = model.get_time()

	n_investigate = 2
	base_size = np.array([6, 4])
	figshape = np.array([n_investigate//2, 2])
	# fig, axes = plt.subplots(n_investigate//2, 2, figsize=figshape*base_size)

	with mpl.rc_context({"font.size": 14}):

		# fig, axes = plt.subplots(n_investigate//2, 2, figsize=figshape*base_size)
		fig, ax = plt.subplots(1, 1, figsize=base_size)

		indices = np.argsort(dx)[-n_investigate:]
		for i in range(n_investigate):
			index = indices[i]
			step_index = np.searchsorted(time, index)
			# ax = axes.ravel()[i]

			n_local = 6
			si, sj = step_index - n_local, step_index + n_local
			chunk = model.cut_index(si, sj+1)

			_data = solver.partition.data.cut_index(time[si], time[sj])

			pwlpartition.simple_model_plot(ax, chunk, _data)
			ax.legend(fontsize=20)

			break

	
# %% 

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	
# ! sampling dx
with mpl.rc_context({"font.size":12, "axes.labelsize": 20}):
	data = local["dx"]
	A, beta = local["right_exp"] 
	fig, axes =  plt.subplots(1, 2, figsize=(12, 4))

	basis = np.linspace(0, data.max(), 1000)
	curve = fit_exp_function(basis, A, beta)

	ax = axes[0]
	sns.histplot(data, ax=ax)
	ax.set_xlabel(r"$\Delta x$")

	fit_label = r"$\exp(-(x - A)/\beta)$"
	ax.plot(basis, curve, c=defcolors[1], linestyle='--', lw=3, label=fit_label)

	legend = ax.legend()
	plt.draw()
	p = legend.get_window_extent()
	
	transform = ax.transAxes.inverted().transform
	x, y = transform(p.p0)
	first_xy = (x, y-0.1)
	ax.annotate(fr'$A = {A:.3f}$, $\beta = {beta:.3f}$', first_xy,
				xycoords='axes fraction')

	second_xy = (x, y-0.2)
	ax.annotate(r'$P(\, \Delta x < 0 ) = {:.3f}$'.format(local["stall_fraction"]), second_xy,
				xycoords='axes fraction')


	ax = axes[1]
	ax.set_xlabel(r"$\Delta x$")
	xlim = (data.min(), np.sort(data)[-10])
	sns.histplot(data, binrange=xlim, ax=ax)
	basis = np.linspace(0, xlim[1], 1000)
	curve = fit_exp_function(basis, A, beta)
	ax.plot(basis, curve, c=defcolors[1], linestyle='--', lw=4)

	

pub.save_figure("candidate_empirical_dx", notename)

# %% 
# !step length

lnbin = 41

with mpl.rc_context(thesis.texstyle):
	data = local["lengths"]
	fig, ax = plt.subplots(figsize=(6,4))
	describe(data)
	xlim = (0, data.max())


	bins = np.linspace(-xlim[1], xlim[1], lnbin, endpoint=True)
	sns.histplot(data, binrange=xlim, bins=lnbin//4+1, ax=ax)
	ly = 1.2 * ax.get_ylim()[1]
	ax.set_ylim((0, ly))

	xy =  (1.1 * 2*candidate_r, 0.9*ly)
	axvpoint = np.array([2*candidate_r, ax.get_ylim()[1]])

	ax.annotate(r'2r', xy)
	ax.plot([2*candidate_r]*2, [0, axvpoint[1]], linestyle='--', c='k')
	# ax.plot(*axvpoint, marker='o', c='k', markersize=6)
	ax.set_xlabel(r"length~(\textmu m)")

	# popt = fit_normal_curve(data)
	# fit_xlim = (-0.1, 1.1*data.max())
	# basis = np.linspace(*fit_xlim, num=1000) 
	# ax.plot(basis, normal_function(basis, *popt), c=defcolors[1], linestyle='--', lw=4)
	# ax.set_xlim(fit_xlim)

	# half_basis = np.linspace(*xlim, num=1000)
	# fill = ax.fill_between(half_basis, normal_function(half_basis, *popt), alpha=0.2, zorder=-1)
	fill = ax.fill_between([0, 2*candidate_r], [100,100], alpha=0.2, zorder=-1)
	# ax.legend([fill], [""])

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))


thesis.save_figure("candidate_length_distribution")
	
# pub.save_figure("candidate_empirical_step_length", notename)
# %% 
np.median(local['lengths'])
np.max(local['lengths'])
# len(local['lengths'])


# %% 
# !deviation angle

with mpl.rc_context(mplstyle):
	# fig, ax = plt.subplots(figsize=(12,8))
	fig, ax = plt.subplots(figsize=(4,4))

	data = local["deviation"]
	describe(data)
	xlim = (-np.pi, np.pi)
	sns.histplot(data, binrange=xlim, ax=ax, stat='density', alpha=0.6)
	xlabel = r"$\theta_{\mathrm{d}}$"
	scale = np.std(data)
	ax.annotate(rf"SD({xlabel}) = {scale:0.2f}", (0.02, 0.9), xycoords='axes fraction', fontsize=15)
	ax.set_xlabel(xlabel)

	ax.set_xticks([-np.pi/2,0,np.pi/2])
	ax.set_xticklabels([r"$-\pi/2$", 0, r"$\pi/2$"])
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	basis = np.linspace(*xlim, num=1000)
	normal = scipy.stats.norm(0, np.std(data))
	# ax.plot(basis, (len(data)/np.pi) * normal.pdf(basis), c=defcolors[1], linestyle='--', lw=4)
	ax.plot(basis, normal.pdf(basis), c=defcolors[1], linestyle='--', lw=5)
	ax.set_ylim(0, 1.2 * normal.pdf(0))


thesis.save_figure("candidate_deviation_angle")
# pub.save_figure("candidate_empirical_deviation_angle", notename)

# %% 
# !segment angles

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots()

	data = local["angles"]
	describe(data)
	print('min absolute angle', np.min(np.abs(data)))
	xlim = (-np.pi, np.pi)
	sns.histplot(data, binrange=xlim, bins=30, ax=ax, stat='count', alpha=0.8)
	xlabel = r"segment--segment angle, $\theta$"
	scale = np.std(data)
	# ax.annotate(rf"std({xlabel}) = {scale:0.3f}", (0.02, 0.9), xycoords='axes fraction', fontsize=15)
	ax.set_xlabel(xlabel)

	ax.set_xticks([-np.pi, -np.pi/2, 0 ,np.pi/2, np.pi])
	ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", 0, r"$\pi/2$", r"$\pi$"])
	ax.set_xlim([-1.1*np.pi, 1.1*np.pi])

	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

thesis.save_figure('segment_angle_distribution')

# %% 

def plot_alpha_scatter(ax, alpha, mstyle={}):
    mscat = {'edgecolor':'white', 'linewidth' : 2.0, 'alpha' : 0.8, 's': 160}
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


with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4,4))

	plot_alpha_scatter(ax, data)

thesis.save_figure('segment_angle_correlation')

print(np.corrcoef(data[:-1], data[1:])[0][1])



# %% 


pub.save_figure("candidate_empirical_segment_angle", notename)


# same but use absolute angle so that the decay at theta = 0 is visible
with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots()

	data = np.abs(local["angles"])
	describe(data)
	print('min absolute angle', np.min(np.abs(data)))
	xlim = (-np.pi, np.pi)
	sns.histplot(data, binrange=xlim, ax=ax, stat='density')
	xlabel = r"$|\theta|$"
	ax.set_xlabel(xlabel)

	ax.set_xticks([-np.pi/2,0,np.pi/2])
	ax.set_xticklabels([r"$-\pi/2$", 0, r"$\pi/2$"])
	# ax.set_ylim(0, 1.8* normal.pdf(0))


pub.save_figure("candidate_empirical_abs_segment_angle", notename)

# %% [markdown]
# its interesting to consider whether the twitching geometry means its unusual
# for the cell to displace immediately in the same direction.
# we can try distinguishing sequential retractions using the mapped velocity
# unfortunately its hard to be convinced that we actually distinguish such retractions with \theta ~ 0
# we might want to use TMOS to see if simulated twitching has this property

# %% 
# * compute the angle correlation between sequential segments, and also for sequntial+1 segments
# we are interested to know if retraction of a single pilus is "broken" by a short displacement in another direction
# todo ...

