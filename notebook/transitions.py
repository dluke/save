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
# generate paper figures on walking to crawling transitions in simulation
# study qualitative crawling/walking transition behaviour

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

import thesis.publication as thesis


import pili
import readtrack
import command
import stats
import parameters
import rtw
import twutils

import pili.publication as pub

notename = 'transitions'
# %% 
# config
# mplconf = {}
# mplconf['fsize'] = 30

plt.rcParams.update({
	'text.usetex': True,
	'axes.labelsize': 30,
	})



# %% 
# mpl config
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
lcolor  = CB91_Blue
mcolor = '#029FE6'
errcolor = '#D4F1FF'

mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['lines.linewidth'] = 2

# setup
def bootstrap(arr, N=10000):
	_bts = [np.random.choice(arr, size=arr.size, replace=True) for _ in range(N)]
	_bts_mean = [np.mean(_b) for _b in _bts]
	std = np.std(_bts_mean)
	# btmean = np.mean(_bts_mean)
	return std

rundir = join(pili.root, "../run")

# %% 
# 1. just vary pilivar and compute walking -> crawling transition time 
_dir = join(rundir, "825bd8f/transition/walking_crawling")
_dc = rtw.DataCube(target=_dir)
_lld = _dc.load_local()
_tau_ls = [np.array(twutils.make_get('simtime.each')(ld)) for ld in _lld]
_tau = [np.mean(arr) for arr in  _tau_ls]

_std = np.array([bootstrap(_arr) for _arr in _tau_ls])
_error = 1.96*_std
_error

basis = _dc.basis[0]
with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(5,5))
	ax.plot(basis, _tau)
	ax.fill_between(basis, _tau-_error, _tau+_error, color=errcolor, ec=lcolor)
	ax.set_ylim((0,1.1*200))
	ax.set_xlabel(r'$\kappa$', fontsize=32)
	ax.set_ylabel('transiton time (s)')
	# ax.set_xlim((20, 150))
	ax.grid(False)
	plt.tight_layout()
	pub.save_figure('w_c_pilivar', notename,  fig=fig)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	# ax.set_xlim(20, None)
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	ax.set_xlim(None, 7.5)


# %% 
_dir = join(rundir, "825bd8f/transition/crawling_walking")
_dc = rtw.DataCube(target=_dir)
_lld = _dc.load_local()
_tau_ls = [np.array(twutils.make_get('simtime.each')(ld)) for ld in _lld]
_tau = [np.mean(arr) for arr in  _tau_ls]

_std = np.array([bootstrap(_arr) for _arr in _tau_ls])
_error = 1.96*_std
_error

# why note ticks/spines ...
basis = _dc.basis[0]
with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(5,5))
	ax.plot(basis, _tau)
	ax.fill_between(basis, _tau-_error, _tau+_error, color=errcolor, ec=lcolor)
	ax.set_ylim((0,1.1*200))
	ax.set_xlabel(r'$\epsilon$', fontsize=32)
	ax.set_ylabel('transiton time (s)')
	ax.set_xlim((20, 150))
	ax.grid(False)
	plt.tight_layout()
	plt.savefig('/home/dan/tmp/notesave.png')
	pub.save_figure('c_w_eps', notename,  fig=fig)

		
	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.set_xlim(20, None)
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))



# %% 

_dir = join(rundir, "825bd8f/transition/wc_tdwell")
_dc = rtw.DataCube(target=_dir)
_lld = _dc.load_local()
_tau_ls = [np.array(twutils.make_get('simtime.each')(ld)) for ld in _lld]
_tau = [np.mean(arr) for arr in  _tau_ls]

_std = np.array([bootstrap(_arr) for _arr in _tau_ls])
_error = 1.96*_std
_error

basis = _dc.basis[0]
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(basis, _tau, color=lcolor, mfc=mcolor)
ax.fill_between(basis, _tau-_error, _tau+_error, color=errcolor, ec=lcolor)
ax.set_ylim((0,1.1*200))
ax.set_xlabel(r'$\tau_{\mathrm{dwell}}$')
ax.set_ylabel('transiton time (s)')
# ax.set_xlim((20, 150))
ax.grid(False)
plt.tight_layout()
pub.save_figure('w_c_tdwell', notename,  fig=fig)


# %% 

_dir = join(rundir, "825bd8f/transition/cw_kspawn")
_dc = rtw.DataCube(target=_dir)
_lld = _dc.load_local()
_tau_ls = [np.array(twutils.make_get('simtime.each')(ld)) for ld in _lld]
_tau = [np.mean(arr) for arr in  _tau_ls]

_std = np.array([bootstrap(_arr) for _arr in _tau_ls])
_error = 1.96*_std
_error

# sns.histplot(_tau_ls[0])

# %% 
basis = _dc.basis[0]
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(basis, _tau, color=lcolor, mfc=mcolor)
ax.fill_between(basis, _tau-_error, _tau+_error, color=errcolor, ec=lcolor)
ax.set_ylim((0,1.1*200))
ax.set_xlabel(r'$k_{\mathrm{spawn}}$')
ax.set_ylabel('transiton time (s)')
# ax.set_xlim((20, 150))
ax.grid(False)
plt.tight_layout()
pub.save_figure('c_w_kspawn', notename,  fig=fig)


# %%
