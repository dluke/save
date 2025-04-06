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
# load our solved models for the high-mean-velocity crawling trajectories
# and post process out the mistakes so we can get some reliable statistics

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

import pili.publication as pub
import thesis.publication as thesis

import pili
from pili import support
import _fj
import mdl
import pwlpartition
import pwltree
import fjanalysis
import pwlstats


# %% 

select_idx = _fj.load_subset_idx()["top"]
look = [join(pwlstats.root, f"run/cluster/no_heuristic/top/_top_{idx:04d}")  for idx in select_idx]
found = [directory for directory in look if os.path.exists(join(directory, "solver.pkl"))]
solverlist= [pwlstats.load_solver_at(directory) for directory in found]

# %% 
# plot the first model

solver = solverlist[1]


n = 400
short = solver.get_model().cut(0,n)
short_data = solver.get_data().cut_index(0,n)

# short = solver.get_model()
# short_data = solver.get_data()

fig, ax, = plt.subplots(figsize=(20,20))
pwlpartition.simple_model_plot(ax, short, short_data)

# plot this but as a partition

# %% 
import itertools

def plot_partition(ax, model, data):

	color = itertools.cycle(['#FEC216', '#F85D66', '#75E76E'])
	time = model.get_time()
	split = np.split(data.get_n2(), time[:None])
	for sp in split:
		c = next(color)
		x, y = sp.T
		ax.plot(x, y, c=c, linestyle='none', marker='o')
	ax.set_aspect("equal")

	ax.set_xlabel('x')
	ax.set_ylabel('y')

fig, ax, = plt.subplots(figsize=(20,20))
plot_partition(ax, short, short_data)
pwlpartition.simple_model_plot(ax, short)


# %% 

c2 = '#DA5025'
c3 = '#2AA1C6'
blue = c3

model_style = {"linestyle": '-', 'lw':2.5, 'alpha':0.6, 'label':'wavelet', 'color':c2, "marker":'D', 'markerfacecolor' : 'none', 'markeredgewidth':2.5, 'markersize': 8}
ptlkw = {"linestyle":'none', 'lw':1.5, "marker":"o", "alpha":0.5, 'color':c3, 'markerfacecolor': 'none', 'markeredgewidth':1.5}

def simple_model_plot(ax, model=None, data=None, model_style=model_style, ptlkw=ptlkw, scale_bar=True):
	
	h2, = ax.plot(data.x, data.y, label='data', **ptlkw)
	marker_only = model_style.copy()
	marker_only["linestyle"] = 'none'
	if model:
		h1, = ax.plot(model.x, model.y, **model_style)
		ax.plot(model.x, model.y, **marker_only)

	if scale_bar:
		plt.draw()
		shift_pt = np.array([0.2,-0.2])
		pt = np.array([ax.get_xlim()[0], ax.get_ylim()[1]]) + shift_pt
		# pt = np.array([0,0.05])
		width = 0.2
		ax.plot([pt[0],pt[0]+width], [pt[1],pt[1]], linewidth=3, c='black', alpha=0.8)
		delta = 0.005
		shift = 0.015
		ax.text(pt[0]+width + delta + shift, pt[1]-delta-shift, r"$~0.2~$\textmu m", fontsize=20)

	ax.axis(False)
	ax.set_aspect('equal')

# %% 

track_index = int(os.path.split(found[1])[-1].split('_')[-1])
ltr = _fj.lintrackload([track_index])[0]
# ltr = ltr.cut(0,400)

time = ltr['time'] - ltr['time'][0]
step_idx_time = time[ltr.step_idx]
n_idx = np.searchsorted(step_idx_time, n/10)

x, y = ltr.x[ltr.step_idx][:n_idx], ltr.y[ltr.step_idx][:n_idx]
xy = mdl.LPtrack(np.arange(x.size), x, y)

# %% 

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(10,10))
	simple_model_style = {**model_style, 'color':defcolor[5]}
	simple_model_plot(ax, xy, short_data, model_style=simple_model_style)

# ax.plot(x, y, **model_style)
# ax.set_aspect('equal')

pub.save_figure("track_3008_cg")

# %%

# ! adjust the sensitivity by eye for this example
short = mdl.recursive_coarsen(short, 0.06)
fig, ax = plt.subplots(figsize=(10,10))
simple_model_plot(ax, test, short_data)

# %%

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(10,10))
	simple_model_plot(ax, short, short_data, scale_bar=False)

	from matplotlib.lines import Line2D
	h1 = Line2D([0], [0], **{**ptlkw, 'alpha':1.0, 'markersize':12})
	h2 = Line2D([0], [0], **{**simple_model_style, 'alpha':1.0})
	h3 = Line2D([0], [0], **{**model_style, 'alpha':1.0})
	ax.legend([h1 , h2, h3], ['data', 'Coarse Grained', 'PWL'], loc="lower right", fontsize=20)

pub.save_figure("track_3008_pwl")

# %% 
model = solver.get_model()
# ! repeat the adjustment on the whole trajectory
model = mdl.recursive_coarsen(model, 0.06)

data = model.get_step_length()
# !remove steps that we visually identify to be mistakes
post_remove_idx = data > 3
post_remove_idx[post_remove_idx.tolist().index(True)-1] =  True
data = data[~post_remove_idx]


xlim = (np.min(data),1.0)
with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	sns.histplot(data, binrange=xlim, alpha=.8, ax=ax, shrink=0.8)
	ax.set_xlabel(r"PWL step distance (\textmu m)")
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	# ax.xaxis.set_major_locator(plt.MaxNLocator(6))
	ax.set_xlim(0,None)
# ax.axvline(solver.sigma)

print("M = ", len(data))
print('median, max', np.median(data), np.max(data))

pub.save_figure("track_3008_step_distance")

# %%
