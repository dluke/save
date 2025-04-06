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
# scale up wavelet_local_map to the whole crawling dataset

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

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

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
notename = "wavelet_em"
notedir = join(pili.root, 'notebook/')

# %% 
wavelet='db1'
em_config = {"wavelet":wavelet, 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}

idx_list = np.loadtxt(join(pili.root, 'notebook/thesis_classification/crawling.npy')).astype(int)
track_list = _fj.trackload_original(idx_list)

def get_data(track):
	wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
	curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr)
	udata = np.diff(curve_coord) 
	return udata, wavemodel, lptr, meta

model_list = [get_data(track) for track in track_list]
udatalist = [m[0] for m in model_list]
metalist = [m[3] for m in model_list]

# %%
# important meta parameter controlling the data preprocessing
q = 0.03
datalist = [emanalyse.asymprocess(udata, q) for udata in udatalist]

# %%
# * LOAD metadata
ldata = fjanalysis.load_summary()
topdata = [ldata[i] for i in idx_list]
vel = np.array([ld['lvel']['mean'] for ld in topdata])

# %%
# * SETUP models

setting = dict(max_iter=1000, tol=1e-6)

def mkgmmrx(n_components, setting=setting):
	def gmmr(setting=setting):
		return sklearn.mixture.GaussianMixture(n_components=n_components, **setting)
	return gmmr

gmmr2, gmmr3, gmmr4 = [mkgmmrx(n) for n in [2,3,4]]
_
names = ['gmmr2', 'gmmr3', 'gmmr4']
# dflist = emanalyse.analyse_mixture(datalist, [gmmr2, gmmr3, gmmr4], names)
dflist = emanalyse.analyse_mixture(datalist, [gmmr2, gmmr3, gmmr4], names)

# %%

cachedir = join(notedir, notename)
if not os.path.exists(cachedir):
	os.makedirs(cachedir)
cache = join(cachedir, 'dflist.pkl')
print('writing to ', cache)
with open(cache, 'wb') as f:
	pickle.dump(dflist, f)


# %%

best_index = [df['bic'].argmin() for df in dflist]
count = collections.Counter(best_index)
tab = [[names[i], count[i]] for i in range(len(count))]
print(tabulate(tab, headers=['model', 'count']))


# %%

gmm2idx = np.argwhere(np.array(best_index) == 0).ravel()
gmm3idx = np.argwhere(np.array(best_index) == 1).ravel()
gmm4idx = np.argwhere(np.array(best_index) == 2).ravel()
gmmidx = [gmm2idx, gmm3idx, gmm4idx]

# %%

bicnorm = np.stack([(df['bic']/df['bic'][1]).to_numpy() for df in dflist])
bicnorm.shape

# %%

# violin plot the mean velocity distributions

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(6,4))

	classvel = [vel[idx] for idx in gmmidx]
	ax.violinplot(classvel)
	ax.xaxis.set_ticks([1,2,3], labels=['gmm2', 'gmm3', 'gmm4'])
	ax.set_ylabel('velocity $\mu m/s$')

	medians = [np.median(v) for v in classvel]
	print([round(m,4) for m in medians])


# %%

style = dict(alpha=0.2)
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(6,4))

	bicn = [bicnorm[i,idx] for i, idx in enumerate(best_index)]
	color = [defcolor[idx] for idx in best_index]
	ax.scatter(vel, bicn, c=color, **style)
	ax.set_ylim((0.975, 1.025))
	ax.set_xlim((0, 0.2))

# %%
# so lets analyse gmm4 trajectories

ct = itertools.count()

# %%

index = next(ct)
print('index', index, 'track_idx', idx_list[index])
data = datalist[[index]]
gmm = sklearn.mixture.GaussianMixture(n_components=4, **setting)
gm = gmm.fit(data.reshape(-1, 1))

print(dflist[gmm4idx[index]]['means'][2].round(5))
# emanalyse.describe(gm, data)

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(6,4))
	emanalyse.quick_plot(ax, gm, data)
	# ax.set_xlim((-0.15, 0.20))


# %%
index, idx_list[index]
# %%
# examine the "forward bias" of this trajectory
u = data

left, right = u[u<0], u[u>0]
nl, nr = len(left), len(right)
print(nl, nr, nr/(nl + nr))


# plot the trajectory

udata, wavemodel, lptr, meta = model_list[index]
fig, ax = plt.subplots(figsize=(200,200))
pwlpartition.simple_model_plot(ax, wavemodel, data=lptr)


