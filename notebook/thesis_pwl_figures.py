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
# create figures for thesis PWL chapter

# %% 
import os
import json
from tqdm import tqdm
import numpy as np
import scipy.stats
import scipy.optimize
import pickle
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd
import itertools
import collections
from tabulate import tabulate

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import thesis.publication as thesis

import pili
from pili import support
import _fj
import mdl
import pwlpartition
import emanalyse

import fjanalysis
import pwlstats

from skimage.restoration import denoise_wavelet
from skimage.restoration import  estimate_sigma

import sklearn.mixture

# %% 
mplstyle = {"font.size": 20}
vstyle = dict(alpha=0.2,c='black',linestyle='--')
notename = "wavelet_local_map"

notedir = join(pili.root, 'notebook/')

# %% 
shstyle = dict(element="step", fill=False, alpha=0.8)
# xlim for plotting
xlim = (-0.08, 0.16)
# xlim for preprocess 
pxlim = (-0.16, 0.14)


# %% 

load_path = join(pili.root, 'notebook/thesis_classification/crawling.npy')
idx_list = np.loadtxt(load_path).astype(int)
track_list = _fj.trackload_original(idx_list)

# %% 
index = 694 # candidate track_idx = 2924
track = track_list[index]

short = track.cut(0,200)
x, y = short['x'], short['y']
_x = x - x.mean()
_y = y - y.mean()
rotate = False
# * cut and flip x,y (rotate 90)
if rotate:
	short['x'] = -1 * _y
	short['y'] = _x

sigma = pwlpartition.estimate_error(track['x'], track['y'])

# %% 
# ! for supplement

c2 = '#DA5025'
c3 = '#2AA1C6'
blue = c3

model_style = {"linestyle": '-', 'lw':2, 'alpha':0.6, 'label':'wavelet', 'color':c2, "marker":'D', 'markerfacecolor' : 'none', 'markeredgewidth':2, 'markersize': 4}
ptlkw = {"linestyle":'none', 'lw':1.5, "marker":"o", "alpha":0.6, 'color':c3, 'markerfacecolor': 'none', 'markeredgewidth':1.5}

def simple_model_plot(ax, model=None, data=None, model_style=model_style, ptlkw=ptlkw, scale_bar=True):
	

	h2, = ax.plot(data.x, data.y, label='data', **ptlkw)
	marker_only = model_style.copy()
	marker_only["linestyle"] = 'none'
	if model:
		h1, = ax.plot(model.x, model.y, **model_style)
		ax.plot(model.x, model.y, **marker_only)


	if scale_bar:
		pt = np.array([0,0.05])
		width = 1.0
		ax.plot([pt[0],pt[0]+width], [pt[1],pt[1]], linewidth=2, c='black', alpha=0.8)
		delta = 0.005
		ax.text(pt[0]+0.1 + delta + 0.005, pt[1]-delta-0.005, r"$0.1$\textmu m", fontsize=14)

	ax.axis(False)
	# ax.legend(fontsize=20, loc=(1.04, 0))
	# ax.legend([h1, h2], [r'curve, $T$', r'data, $\bm{x}$'], fontsize=12)
	# ax.legend([h1, h2, h3], [r'curve, $T$', r'data, $\bm{x}$', r'map, $T(s)$'], fontsize=12)

	ax.set_aspect('equal')


# wavelet transform these data
wavelet='db1'
em_config = {"wavelet":wavelet, 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}

# just wavlet, no coarsening
def local_wave_guess(track, config=em_config, sigma=None):

	def estimate_error(track):
		x, y = track['x'], track['y']
		return np.mean([estimate_sigma(x), estimate_sigma(y)])


	# 
	_x = track["x"]
	_y = track["y"]
	if sigma == None:
		sigma = estimate_error(track)
	x = denoise_wavelet(track["x"], **config)
	y = denoise_wavelet(track["y"], **config)
	lptr = mdl.LPtrack(None, _x, _y)
	lptr.reset_dt()
	wavemodel = mdl.LPtrack(None, x, y)
	wavemodel.reset_dt()
	meta = {"sigma" : sigma}
	return wavemodel, lptr, meta

# wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
# wavemodel, lptr, meta = local_wave_guess(track, config=em_config)
print('using sigma = ', sigma)
wavemodel, lptr, meta = local_wave_guess(short, config=em_config, sigma=sigma)

def coarsen(model, sigma):
	denoised = np.stack([model.x, model.y])
	coarse_model = pwlpartition.model_from_denoised(denoised, sigma)
	return coarse_model

coarse_model = coarsen(wavemodel, sigma)

# fig, ax = plt.subplots(figsize=(10,10))

column_width = 3.385
f_size = np.array([column_width, column_width])
use_size = 2*f_size
# use_size = 2*np.array([12,12])

texstyle = {
	"font.size": 10, "ytick.labelsize": 9, "xtick.labelsize": 9, 
	"text.usetex": True, "axes.labelsize" : 10, "legend.frameon" : False,
	"xtick.direction": 'in', "ytick.direction": 'in'
	} 
usestyle = texstyle
plt.rc('text.latex', preamble=r"\usepackage{bm}")

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=use_size)
	simple_model_plot(ax, model=None, data=lptr, scale_bar=False)

	from matplotlib.lines import Line2D
	h1 = Line2D([0], [0], **{**ptlkw, 'alpha':1.0})
	h2 = Line2D([0], [0], **{**model_style, 'alpha':1.0})
	h3 = Line2D([0], [0], **{**_m_style, 'alpha':1.0})
	h4 = Line2D([0], [0], **{**_pwl_style, 'alpha':1.0})
	labels = ["data", "wavelet", "processed wavelet", "PWL"]
	handles = [h1, h2, h3, h4]
	ax.legend(handles, labels, loc=(0.4,0.6), fontsize=14)

	pt = np.array([lptr.x[0], lptr.y[0] + 0.5])
	width = 0.2
	ax.plot([pt[0],pt[0]+width], [pt[1],pt[1]], linewidth=3, c='black', alpha=0.8)
	delta = 0.005
	ax.text(pt[0]+width + delta + 0.060, pt[1]-delta-0.030, r"$0.2$~\textmu m", fontsize=16)


thesis.save_figure('pwl_trajectory_data')

_ptlkw = ptlkw.copy()
_ptlkw["alpha"] = 0.4


with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=use_size)
	simple_model_plot(ax, wavemodel, data=lptr, ptlkw=_ptlkw, scale_bar=False)

thesis.save_figure('pwl_trajectory_db1')

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=use_size)
	_m_style = model_style.copy()
	_m_style.update(dict(markersize=7))
	simple_model_plot(ax, coarse_model, data=lptr, model_style=_m_style, ptlkw=_ptlkw, scale_bar=False)

thesis.save_figure('pwl_trajectory_db1_processed')

# !load the candidate solution

target = "/home/dan/usb_twitching/sparseml/run/partprio/_candidate_pwl"
solver = pwlstats.load_solver_at(target)
pwl = solver.get_model()
short_pwl = pwl.cut(0,200)

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=use_size)
	_pwl_style = model_style.copy()
	_pwl_style.update(dict(markersize=7, color='purple'))
	simple_model_plot(ax, short_pwl, data=lptr, model_style=_pwl_style, ptlkw=_ptlkw, scale_bar=False)


thesis.save_figure('pwl_trajectory_pwl')


# %% 
model = solver.get_model()

sns.histplot( model.get_distance() )

# %% 
# ! not required

# # %% 
# # sliding window average trajectory

# def sliding_window_guess(track, n):
# 	_x = track['x']
# 	_y = track['y']
# 	lptr = mdl.LPtrack(None, _x, _y)
# 	lptr.reset_dt()

# 	def pad(x):
# 		return np.pad(x, n//2, 'constant', constant_values=np.nan)

# 	N = len(track)
# 	x = np.array([np.nanmean(pad(_x)[i:i+n]) for i in range(N)])
# 	y = np.array([np.nanmean(pad(_y)[i:i+n]) for i in range(N)])

# 	mean_model = mdl.LPtrack(None, x, y)
# 	# mean_model.reset_dt()

# 	return mean_model, lptr

# for n in [5,10]:
# 	mean_model, lptr = sliding_window_guess(short, n=n)
# 	with mpl.rc_context(texstyle):
# 		fig, ax = plt.subplots(figsize=use_size)
# 		simple_model_plot(ax, mean_model, data=lptr)

# 	pub.save_figure('trajectory_mean_model')

# with mpl.rc_context(texstyle):
# 	fig, ax = plt.subplots(figsize=use_size)

# 	_m_style = model_style.copy()
# 	_m_style.update(dict(markersize=7))

# 	simple_model_plot(ax, coarsen(mean_model, sigma), data=lptr, model_style=_m_style, scale_bar=False)
# 	pub.save_figure('trajectory_mean_model_processed')
