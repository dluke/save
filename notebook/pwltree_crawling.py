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
# task a computer with using pwltree to compute fast PWL approximations of the crawling trajectory data


# %% 
import os
import json
import numpy as np
import scipy.stats
import scipy.optimize
import pickle
import itertools
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd

import sctml
import sctml.publication as pub
pub.set_write_dir("/home/dan/usb_twitching/sparseml/EM_paper/impress")
print("writing figures to", pub.writedir)

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import pili
from pili import support
import twutils
import _fj
import fjanalysis
import mdl
import emanalyse

import readtrack
import command

import pwlpartition
import pwltree

# %% 
mplstyle = {"font.size": 20}
# notename = "simulated_em"
# plotting = True

# defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']

# %%
# load the metadata aswell
load_path = join(pili.root, 'notebook/thesis_classification/crawling.npy')
idx_list = np.loadtxt(load_path).astype(int)
track_list = _fj.trackload_original(idx_list)

ldata = fjanalysis.load_summary()
topdata = [ldata[i] for i in idx_list]
vel = np.array([ld['lvel']['mean'] for ld in topdata])
print(vel.size)

vel_order = np.argsort(vel)
vel_order[-1]

vel.min(), vel.max()

# %% 
# get the median error estimate

sigma_list = [estimate_sigma(track) for track in track_list]
'sigma', np.median(sigma_list)
# %% 
min(sigma_list), np.median(sigma_list), max(sigma_list)
pwlpartition.estimate_r(np.median(sigma_list))

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(vel, sigma_list, alpha=0.1)
ax.set_ylim(0, 0.02)


# %% 

def pwl_tree_solve(data, r):
	tsolver = pwltree.TreeSolver(data, overlap=True)
	tsolver.build_max_tree()
	tsolver.build_priority()
	tsolver.solve(pwltree.stop_at(r))
	return tsolver

def get_lptr(track):
	dt = np.insert(np.diff(track['time']), 0, 0)
	lptr = mdl.LPtrack(dt, track['x'], track['y'])
	return lptr

def estimate_sigma(track):
	x, y = track['x'], track['y']
	return pwlpartition.estimate_error(x, y)

track = track_list[0]
track = track_list[650]
r_stop = pwlpartition.estimate_r(estimate_sigma(track))
print('r', r_stop)
track_data = get_lptr(track) #! use mtrack
N_short = 800
short_data = track_list[0].cut(0, N_short)
short_track = track.cut(0,N_short)

def pwl_annealing(track):
	lptr = mdl.get_lptrack(track)

	wavemodel, lptr, meta = pwlpartition.initial_guess(lptr.x, lptr.y)

	loss_conf = {}
	partition = pwlpartition.PartAnneal(wavemodel, lptr, loss_conf=loss_conf, inter_term=True)
	rng = np.random.RandomState()
	solver = pwlpartition.Solver(partition, rng=rng, r=meta["r"], sigma=meta["sigma"], min_constraint=1)
	control = {'maxiter': 10000, 't_end': 0., 'tolerance': 1e-6}

	with support.PerfTimer() as timer:
		solver.linear_solve()
		solver.percolate()
		solver.percolate()

	# output = pwlpartition.Output(target_dir, allow_write=True, index=trackidx)
	with support.PerfTimer() as timer:
		solver.priority_solve(control, output=None)
	return solver
	

# pwlpartition solver
with support.Timer():
	anneal = pwl_annealing(short_track)

	
# pwl solver
# with support.Timer():
# 	tree_solver = pwl_tree_solve(track_data, r_stop)

# %% 
# ! save annealing solution here if we don't want to overwrite it

m = anneal.get_model()
d = anneal.get_data()

_fsize = (20,20)
fig, ax = plt.subplots(figsize=_fsize)
pwlpartition.simple_model_plot(ax, m, d)


# %% 

# data_dir = join(pili.root, 'notebook/pwltree/crawling')
data_dir = join(pili.root, 'notebook/pwltree/annealing/')
def load_pwl(path):
	with open(path, 'rb') as f:
		return pickle.load(f)
	
solver = load_pwl(join(data_dir, 'pwl_2751.pkl'))



# def plot_solver(ax, solver):
# 	model = solver.get_model()
# 	data = mdl.LPtrack(None, *solver.get_data().T)
# 	pwlpartition.simple_model_plot(ax, model, data)

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1, c2, c3 = defcolors[:3] 
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.3, 'color':c3}
model_style = {"linestyle": '-', 'lw':5, 'alpha':0.8, 'label':'model', 'color':c2, "marker":'D'}

def simple_model_plot(ax, model, data=None, pwl=None, 
		ptlkw=ptlkw, model_style=model_style):
	
	truth_style = {"linestyle": '--', 'lw':2, 'alpha':0.5, 'label':'truth', 'color':c1}

	if pwl:
		ax.plot(pwl.x, pwl.y, **truth_style)

	if data:
		ax.plot(data.x, data.y, label='data', **ptlkw)

	ax.plot(model.x, model.y, **model_style)

	ax.set_aspect('equal')
	# ax.legend(fontsize=20, loc=(1.04, 0))

def plot_solver(solver, size_factor=20, cut=None, axis_switch='off', 
		ptlkw=ptlkw, model_style=model_style):

	model = solver.get_model()
	data = solver.get_data()
	c = data.center()
	model.center(at=c)
	if cut != None:
		data = data.cut_index(0,cut)
		model = model.cut(0,cut)

	sx, sy = model.get_bounding_size(buffer=(0.1,0.1))
	_fsize = size_factor * np.array([sx, sy])
	fig, ax = plt.subplots(figsize=_fsize)
	ax.axis(axis_switch)

	# TODO annotate a ruler

	simple_model_plot(ax, model, data, ptlkw=ptlkw, model_style=model_style)
	return fig, ax

plot_solver(solver, cut=50)


# %% 
# * Load the annealing data
list_dir = sorted([join(data_dir, l) for l in os.listdir(data_dir)])

def get_index(path):
	return int(os.path.splitext(list_dir[0])[0].split('_')[-1])

# for path in list_dir:
# 	index = get_index(path)
# 	solver = load_pwl(path)
# 	model = solver.get_model()

solver_list = [load_pwl(path) for path in list_dir]
# %% 
load_idx = np.array([solver.track_index for solver in solver_list])
model_list = [solver.get_model() for solver in solver_list]

# get the lengths

lengths = [model.get_distance() for model in model_list]
median_length = [np.median(l) for l in lengths]
mean_length = [np.mean(l) for l in lengths]

texstyle = {"font.size": 20, "text.usetex": True, "axes.labelsize" : 24}
with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	scatstyle = {'edgecolor':'white', 'linewidth' : 1.0, 'alpha' : 0.5, 's': 60}
	ax.scatter(vel[:len(load_idx)], median_length, **scatstyle)
	# ax.scatter(vel[:len(load_idx)], mean_length, **scatstyle)
	
	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	ax.set_xlabel(r"mean velocity $(\mu m/s)$")
	ax.set_ylabel(r"median length $(\mu m)$")
	# ax.set_ylabel(r"median length $(\mu m)$")
	ax.set_xlim(0,None)
	ax.set_ylim(0,None)


pub.save_figure("pwl_mean_velocity_median_length")

# %% 
# pick a fast and slow trajectory and plot them
i_vel = sorted(list(enumerate(vel[:len(load_idx)])), key=lambda t: t[1])

slow_idx = i_vel[3][0]
# slow_track_idx = 4
# fast_track_idx = 1187

s1 = solver_list[slow_idx]

fig, ax = plot_solver(s1, axis_switch='off')

rect = mpl.patches.Rectangle((-0.2, -0.1), 0.2, 0.01, facecolor='black', alpha=0.75)
ax.add_patch(rect)


pub.save_figure('slow_pwl')

# %% 

fast_idx = i_vel[227][0]
s2 = solver_list[fast_idx]

_ptlkw = ptlkw.copy()
_ptlkw['markersize'] = 20
_ptlkw['lw'] = 8

_model_style = model_style.copy()
_model_style['lw'] = 12


fig, ax = plot_solver(s2, cut=150, axis_switch='off', ptlkw=_ptlkw, model_style=_model_style)
# ax.plot([0,0], [1,1], transform=ax.transAxes)

rect = mpl.patches.Rectangle((-1.0, 1.8), 0.2, 0.01, facecolor='black', alpha=0.75)
ax.add_patch(rect)

pub.save_figure('fast_pwl')

print('mean velocities', vel[slow_idx], vel[fast_idx], '\mu m/s')


# %% 
