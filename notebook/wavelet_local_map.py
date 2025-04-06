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
# For estimating local speed distribution / trajectory shape
#  use wavelet transform rather than PWL method

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
# pub.set_write_dir("/home/dan/usb_twitching/sparseml/EM_paper")
# pub.set_write_dir("/home/dan/usb_twitching/sparseml/EM_paper/impress")
pub.set_write_dir("/home/dan/usb_twitching/sparseml/EM_paper/prx")
print("writing figures to", pub.writedir)

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

track = track_list[766]

# * cut and flip x,y (rotate 90)
short = track.cut(0,200)
x, y = short['x'], short['y']
_x = x - x.mean()
_y = y - y.mean()
short['x'] = -1 * _y
short['y'] = _x

sigma = pwlpartition.estimate_error(track['x'], track['y'])

# %% 
# ! for supplement

c2 = '#DA5025'
c3 = '#2AA1C6'
blue = c3

model_style = {"linestyle": '-', 'lw':2, 'alpha':0.6, 'label':'wavelet', 'color':c2, "marker":'D', 'markerfacecolor' : 'none', 'markeredgewidth':2, 'markersize': 3}
ptlkw = {"linestyle":'none', 'lw':1.5, "marker":"o", "alpha":0.6, 'color':c3, 'markerfacecolor': 'none', 'markeredgewidth':1.5}

def simple_model_plot(ax, model, data, model_style=model_style, ptlkw=ptlkw, scale_bar=True):
	

	h1, = ax.plot(model.x, model.y, **model_style)
	h2, = ax.plot(data.x, data.y, label='data', **ptlkw)
	marker_only = model_style.copy()
	marker_only["linestyle"] = 'none'
	ax.plot(model.x, model.y, **marker_only)


	if scale_bar:
		pt = np.array([0,0.05])
		ax.plot([pt[0],pt[0]+0.1], [pt[1],pt[1]], linewidth=2, c='black', alpha=0.8)
		delta = 0.005
		ax.text(pt[0]+0.1 + delta + 0.005, pt[1]-delta-0.005, r"$0.1$\textmu m", fontsize=14)

	ax.axis(False)
	# ax.legend(fontsize=20, loc=(1.04, 0))
	ax.legend([h1, h2], [r'curve, $T$', r'data, $\bm{x}$'], fontsize=12)
	# ax.legend([h1, h2, h3], [r'curve, $T$', r'data, $\bm{x}$', r'map, $T(s)$'], fontsize=12)

	ax.set_aspect('equal')


# wavelet transform these data
wavelet='db2'
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

texstyle = {
	"font.size": 10, "ytick.labelsize": 9, "xtick.labelsize": 9, 
	"text.usetex": True, "axes.labelsize" : 10, "legend.frameon" : False,
	"xtick.direction": 'in', "ytick.direction": 'in'
	} 
usestyle = texstyle
plt.rc('text.latex', preamble=r"\usepackage{bm}")

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=use_size)
	simple_model_plot(ax, wavemodel, data=lptr)

pub.save_figure('trajectory_db2')

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=use_size)
	_m_style = model_style.copy()
	_m_style.update(dict(markersize=7))
	simple_model_plot(ax, coarse_model, data=lptr, model_style=_m_style, scale_bar=False)

pub.save_figure('trajectory_db2_processed')



# %% 

_wavemodel, lptr, meta = local_wave_guess(track, config=em_config, sigma=sigma)
disp = np.sqrt(np.diff(_wavemodel.x)**2 + np.diff(_wavemodel.y)**2)
print('zeros', (disp==0).sum()/len(disp))

track = track_list[501]

xlim = (0, 0.02)
shstyle = dict(element="step", stat='density', fill=False, alpha=0.6, lw=3)
with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=f_size/2)
	sns.histplot(disp, log_scale=False, ax=ax, binrange=xlim, **shstyle)
	ax.set_xlabel(r"displacement (\textmu m)")

	ax.yaxis.set_major_locator(plt.MaxNLocator(5))

	ax.set_ylabel('')
	ax.yaxis.set_ticks([])
	ax.set_ylim(0, 280)

pub.save_figure("db2_displacement_distribution")

# %% 
# sliding window average trajectory

def sliding_window_guess(track, n):
	_x = track['x']
	_y = track['y']
	lptr = mdl.LPtrack(None, _x, _y)
	lptr.reset_dt()

	def pad(x):
		return np.pad(x, n//2, 'constant', constant_values=np.nan)

	N = len(track)
	x = np.array([np.nanmean(pad(_x)[i:i+n]) for i in range(N)])
	y = np.array([np.nanmean(pad(_y)[i:i+n]) for i in range(N)])

	mean_model = mdl.LPtrack(None, x, y)
	# mean_model.reset_dt()

	return mean_model, lptr

for n in [5,10]:
	mean_model, lptr = sliding_window_guess(short, n=n)
	with mpl.rc_context(texstyle):
		fig, ax = plt.subplots(figsize=use_size)
		simple_model_plot(ax, mean_model, data=lptr)

	pub.save_figure('trajectory_mean_model')

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=use_size)

	_m_style = model_style.copy()
	_m_style.update(dict(markersize=7))

	simple_model_plot(ax, coarsen(mean_model, sigma), data=lptr, model_style=_m_style, scale_bar=False)
	pub.save_figure('trajectory_mean_model_processed')


# %% 
n = 10

track = track_list[501]

_mean_model, lptr = sliding_window_guess(track, n=n)
disp = np.sqrt(np.diff(_mean_model.x)**2 + np.diff(_mean_model.y)**2)
print('zeros', (disp==0).sum()/len(disp))

xlim = (0, 0.02)
shstyle = dict(element="step", stat='density', fill=False, alpha=0.6, lw=3)

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=f_size/2)
	sns.histplot(disp, log_scale=False, ax=ax, binrange=xlim, **shstyle)
	ax.set_xlabel(r"displacement (\textmu m)")
	ax.yaxis.set_major_locator(plt.MaxNLocator(5))
	ax.set_ylim(0, 260)

pub.save_figure("meanmodel_displacement_distribution")



# %% 
# * DEFINE the local distance
# a sensible definition would be the shortest distance to the three adjacent
def estimate_linear_coord(track):
	wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
	return compute_curve_coordinate(wavemodel, lptr)

def compute_curve_coordinate(wavemodel, lptr):
	time = wavemodel.get_time().astype(int)

	M = len(wavemodel)
	N = len(lptr)
	step_length = np.insert(wavemodel.get_step_length(), 0, 0)

	ab = wavemodel.get_n2()
	pdata = lptr.get_n2()

	# create the local distance matrix
	ln  = 100
	local = np.full((N, ln), np.inf)
	# create curve coordinate 
	curve = np.full((N, ln), np.inf)

	# for each segment
	curve_current = 0

	for i in range(M-1):
		curve_current += step_length[i]
		# get adjacent segments
		a = ab[i].reshape(1,2)
		b = ab[i+1].reshape(1,2)

		# clip indices
		# for k in [-1, 0, 1]:
		for k in np.arange(-ln//2, ln//2).astype(int):
			il, ir = i + k, i + k + 1
			if il < 0 or ir >= M:
				continue
			lidx, ridx = time[il], time[ir]
			local_data = pdata[lidx:ridx]
			s, dist = support.line_coord_dist(local_data, a, b)
			local[lidx:ridx,k+1] = dist
			curve_coord =  s + curve_current
			curve[lidx:ridx,k+1] = curve_coord

	# reduce, keeping track of the curve coordinate

	local.shape
	midx = np.argmin(local, axis=1)
	curve_coord = np.array([curve[i, midx[i]] for i in range(len(curve))])
	return curve_coord

curve_coord = compute_curve_coordinate(wavemodel, lptr)
udata = np.diff(curve_coord)
# udata

def asymprocess(udata, q, side='both'):
	# print('debug: ', udata.size, q)
	zdata = udata[udata!=0]
	ldata, rdata = zdata[zdata<0], zdata[zdata>=0]
	xlim = np.quantile(ldata, q), np.quantile(rdata, 1-q)
	if side == 'both':
		data = zdata[ np.logical_and(zdata > xlim[0], zdata < xlim[-1]) ]
	elif side == 'right':
		data = zdata[zdata < xlim[-1]]
	return data

# data = asymprocess(udata, q)

# %%

# setting = dict(max_iter=1000, tol=1e-8)
# def fit(data, means_init):
# 	gmm = sklearn.mixture.GaussianMixture(n_components=len(means_init), means_init=means_init, **setting)
# 	gm = gmm.fit(data.reshape(-1, 1))
# 	return gm

# def fixed_process(data, xlim=(-0.1, 0.10), delta=0.005):
# 	data = data[np.isfinite(data)]
	
# 	data = emanalyse.remove_pairs(data ,delta=delta)
# 	rs = data.size
# 	keep = np.logical_and(data > xlim[0], data < xlim[1])
# 	return data[keep]


# data = asymprocess(udata, 0.03)

# # means_init = np.array([0.0, 0.03])[:, np.newaxis]
# means_init = np.array([-0.03, 0.0, 0.03])[:, np.newaxis]
# gm = fit(data, means_init)

# print(emanalyse.describe(gm, data))

# with mpl.rc_context({'font.size': 16}):
# 	fig, ax = plt.subplots(figsize=(6,4))
# 	emanalyse.quick_plot(ax, gm, data)
# 	ax.set_xlim((-0.05, 0.10))


# %%
# so fit both n_component = 2/3 systems on the whole dataset
# * LOADING

from glob import glob
look = glob( join(pwlstats.root, f"run/cluster/no_heuristic/top/_top_*") )
found = [directory for directory in look if os.path.exists(join(directory, "solver.pkl"))]
idx_list = [int(name.split('_')[-1]) for name in found]
track_list = _fj.trackload_original(idx_list)

def get_data(track):
	wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
	curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr, ln=11)
	udata = np.diff(curve_coord) 
	return udata, wavemodel, lptr, meta

model_list = [get_data(track) for track in tqdm(track_list)]
udatalist = [m[0] for m in model_list]
metalist = [m[3] for m in model_list]

# important meta parameter controlling the data preprocessing
q = 0.03
datalist = [asymprocess(udata, q) for udata in udatalist]

# %%

def clear_nan(data):
	rmidx = np.logical_or(np.isnan(data),  np.isinf(data))
	return data[~rmidx]


# %%
_xlim = (-0.05, 0.10)

init2 = np.array([0.0, 0.03])[:, np.newaxis]
init3 = np.array([-0.03, 0.0, 0.03])[:, np.newaxis]

index = 4
udata = clear_nan(udatalist[index])
data = datalist[index]
data = emanalyse.symprocess(udata, 0.005)
save_data = data
# data = emanalyse.remove_pairs(emanalyse.symprocess(udata, 0.005))
gm = fit(data, init3)

print('meta', metalist[index])
emanalyse.describe(gm, data)


def pub_context_plot(ax, mm, data, separate=True, no_model=False):
	shstyle = dict(element="step", stat='density', fill=False, alpha=0.6, lw=4)
	defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
	partstyle = dict(alpha=0.7, lw=3)

	sns.histplot(data, ax=ax, color=defcolor[0], **shstyle)
	h1 = ax.lines[-1]

	xspace = np.linspace(data.min(), data.max(), 1000)

	if separate:
		color = itertools.cycle([defcolor[2]])
	else:
		color = itertools.cycle(defcolor)

	it = zip(gm.means_.ravel(), gm.covariances_.ravel(), gm.weights_.ravel())
	if no_model:
		it = []
		h2 = None

	for mean, var, weight in it:
		mm = weight * scipy.stats.norm(mean, np.sqrt(var)).pdf(xspace)
		h2,  = ax.plot(xspace, mm, color=next(color), **partstyle)

	
	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	ax.set_ylabel(r"Density")
	ax.set_xlabel(r"displacement (\textmu m)")

	return [h1, h2]


column_width = 3.385
f_size = (column_width, column_width)
fh_size = 1.0 * np.array([column_width/2, column_width/2])
use_size = fh_size
# use_size = (6,6)

texstyle = {
	"font.size": 10, "ytick.labelsize": 9, "xtick.labelsize": 9, 
	"text.usetex": True, "axes.labelsize" : 10, "legend.frameon" : False,
	"xtick.direction": 'in', "ytick.direction": 'in'
	} 
usestyle = texstyle

texstyle = thesis.texstyle
usesize = (4,4)

# with mpl.rc_context({'font.size': 20}):
with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=usesize)
	vstyle = dict(alpha=0.3,c='black',linestyle='--', lw=2)
	ax.axvline(-0.03, **vstyle)
	ax.axvline(0.03, **vstyle)
	handles = pub_context_plot(ax, gm, data, no_model=False)
	ax.set_xlim(_xlim)
	# ax.legend(handles, ["data", "GMM3"], fontsize=18, loc='upper right')

	ax.annotate(r'$\pm 0.03$', xy=(-0.03, 7), xytext=(0.05, 50), 
		arrowprops = dict(facecolor ='grey', shrink = 0.01, alpha=.5))
	ax.annotate('', xy=(0.03, 14), xytext=(0.05, 50), 
		arrowprops = dict(facecolor ='grey', shrink = 0.01, alpha=.5))

pub.save_figure('example_sus_peaks')
thesis.save_figure('example_sus_peaks')

# %%


udata = clear_nan(udatalist[index])
data = emanalyse.remove_pairs(emanalyse.symprocess(udata, 0.005))
print('fractions', save_data.size, data.size, (save_data.size-data.size)/udata.size)

gm = fit(data, init3)

emanalyse.describe(gm, data)

_texstyle = texstyle.copy()
_texstyle['legend.frameon'] = True
with mpl.rc_context(_texstyle):
	fig, ax = plt.subplots(figsize=usesize)
	ax.axvline(-0.03, **vstyle)
	ax.axvline(0.03, **vstyle)
	# r1 = mpl.patches.Rectangle((-0.04, 0), 0.02, 15, alpha=0.2)
	# ax.add_patch(r1)
	# r1 = mpl.patches.Rectangle((0.02, 0), 0.02, 15, alpha=0.2)
	# ax.add_patch(r1)
	
	handles = pub_context_plot(ax, gm, data, no_model=False)
	ax.set_xlim(_xlim)

	ax.set_ylabel('')
	ax.set_yticks([])

	# from matplotlib.ticker import ScalarFormatter
	# sf = ScalarFormatter(useOffset=False)
	# sf.set_scientific(True)
	# sf.set_powerlimits((0,0))
	# ax.xaxis.set_major_formatter( sf )

	# ax.set_xlabel(r"X $(\mu m)$")

	# ax.legend(handles, ["data", "GMM3"], loc='upper right', framealpha=1.0)
	ax.legend(handles, ["data", "GMM3"], loc=(0.6, 0.6), framealpha=1.0, handlelength=1)
	# ax.yaxis.set_offset_position('right')

# ax.xaxis.get_offset_text().set_position((0,0))

pub.save_figure('sus_example_remove_pairs')
thesis.save_figure('sus_example_remove_pairs')

# %%
# it seems that we can't avoid comparing PWL and wavelet distributions?
# TODO

# %%
# * SCALE UP
# run over the whole dataset and record which model is better and the parameters

def gmmr2():
	return sklearn.mixture.GaussianMixture(n_components=2, **setting)

def gmmr3():
	return sklearn.mixture.GaussianMixture(n_components=3, **setting)

def gmmr4():
	return sklearn.mixture.GaussianMixture(n_components=4, **setting)

# %%

names = ['gmmr2', 'gmmr3', 'gmmr4']
dflist = emanalyse.analyse_mixture(datalist, [gmmr2, gmmr3, gmmr4], names)


# %%

dflist = [emanalyse.regularise(df) for df in dflist]

# %%
def bicnorm(df):
	# add the BIC normalised against gmm3 as a dataframe column
	df = df.copy()
	bic = df['bic']
	df['bicnorm'] = bic/bic[1]
	return df

dflist = [bicnorm(df) for df in dflist]

# %%

out = join(notedir, notename, "dflist.pkl")
if not os.path.exists(out):
	os.makedirs(out)
print('writing to ', out)
with open(out, 'wb') as f:
	pickle.dump(dflist, f)


# %%
from IPython.display import display, HTML
def listfm(l):
	return '[' + ','.join(['{:.4f}'.format(x) for x in l]) + ']'

formatter = {
	'means' : listfm,
	'std' : listfm,
	'weights' : listfm
}
display(HTML(dflist[0].to_html(formatters=formatter)))
def gmdisplay(df):
	display(HTML(df.to_html(formatters=formatter)))


# %%

best_index = [df['bic'].argmin() for df in dflist]
count = collections.Counter(best_index)
tab = [[names[i], count[i]] for i in range(len(count))]
print(tabulate(tab, headers=['model', 'count']))

	

# %%
gmdisplay( dflist[4] )

# %%

df2 = [df for i, df in enumerate(dflist) if best_index[i] == 0]
df3 = [df for i, df in enumerate(dflist) if best_index[i] == 1]
df4 = [df for i, df in enumerate(dflist) if best_index[i] == 2]
print('selecting {} trajectories'.format(len(df3)))

# %%

modelidx = 1
m_, m0, mp = list(zip(*[df['means'][modelidx] for df in df3]))

with mpl.rc_context({'font.size': 12}):
	fig, ax = plt.subplots(figsize=(6,4))
	ax.violinplot([m_, m0, mp], [-1, 0, 1])

	ax.xaxis.set_ticks([-1,0,1], labels=["-ve", "0", "+ve"])
	ax.set_ylabel('displacement $(\mu m$)')
	ax.set_ylim((-0.1,0.1))

	median= [np.median(m) for m in [m_, m0, mp]]
	for m in median:
		ax.axhline(m, **vstyle)
	print('median', [round(m, 6) for m in median])


# %%

w_, w0, wp = list(zip(*[df['weights'][modelidx] for df in df3]))

with mpl.rc_context({'font.size': 12}):
	fig, ax = plt.subplots(figsize=(6,4))
	ax.violinplot([w_, w0, wp], [-1, 0, 1])

	ax.xaxis.set_ticks([-1,0,1], labels=["-ve", "0", "+ve"])
	ax.set_ylabel('weight')
	# ax.set_ylim((-0.1,0.1))

	median= [np.median(m) for m in [w_, w0, wp]]
	for m in median:
		ax.axhline(m, **vstyle)
	print('median', [round(m, 6) for m in median])


# %%

v_, v0, vp = list(zip(*[df['std'][modelidx] for df in df3]))

with mpl.rc_context({'font.size': 12}):
	fig, ax = plt.subplots(figsize=(6,4))
	ax.violinplot([v_, v0, vp], [-1, 0, 1])

	ax.xaxis.set_ticks([-1,0,1], labels=["-ve", "0", "+ve"])
	ax.set_ylabel('$\sigma$', fontsize=20)
	# ax.set_ylim((-0.1,0.1))

	median= [np.median(m) for m in [v_, v0, vp]]
	for m in median:
		ax.axhline(m, **vstyle)
	print('median', [round(m, 6) for m in median])
	ax.set_ylim(0, 0.05)

# %%
# * PLOT
# plot example df2/df4 trajectories
gmm2idx = np.argwhere(np.array(best_index) == 0).ravel()

index = gmm2idx[2]

data = datalist[index]
gmm = sklearn.mixture.GaussianMixture(n_components=2, **setting)
gm = gmm.fit(data.reshape(-1, 1))

emanalyse.describe(gm, data)

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(6,4))
	emanalyse.quick_plot(ax, gm, data)
	ax.set_xlim((-0.15, 0.20))


# %%

gmm4idx = np.argwhere(np.array(best_index) == 2).ravel()
count = itertools.count()

# %%
index = gmm4idx[next(count)]

print('index', index, 'track_index', idx_list[index])

data = datalist[index]
gmm = sklearn.mixture.GaussianMixture(n_components=4, **setting)
gm = gmm.fit(data.reshape(-1, 1))

emanalyse.describe(gm, data)

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(6,4))
	emanalyse.quick_plot(ax, gm, data)
	ax.set_xlim((-0.15, 0.20))


# %%
# * LOAD metadata
ldata = fjanalysis.load_summary()
topdata = [ldata[i] for i in idx_list]
vel = np.array([ld['lvel']['mean'] for ld in topdata])

# %%

# when gmm4 is estimated to be the best model, is gmm3 good enough?
# compare this to the difference between gmm2 and gmm3 
r0, r2 = np.median([df['bicnorm'][0] for df in df2]), np.median([df['bicnorm'][2] for df in df4])
print(r0, r2, (1-r0)/(1-r2) )

# TODO 
def compute_absr2(bic):
	b2, b3 = bic[0], bic[1]
	return abs(b3 - b2)/abs(b2 + b3)
	
def compute_absr4(bic):
	b3, b4 = bic[1], bic[2]
	return abs(b3 - b4)/abs(b3 + b4)

absr2 = np.array([compute_absr2(df['bic']) for df in dflist])
absr4 = np.array([compute_absr4(df['bic']) for df in dflist])

print(np.median(absr2))
print(np.median(absr4))

# and specifically where gmm4 and gmm2 are the best models

# print(np.median(absr2[gmm2idx]))
# print(np.median(absr4[gmm4idx]))

# %%

# plot mean velocity vs. BIC (normalised against gmm3)
# TODO does this start to change at lower mean velocities?

style = dict(alpha=0.2)
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(6,4))

	bicn = [df['bicnorm'][idx] for df, idx in zip(dflist, best_index)]
	color = [defcolor[idx] for idx in best_index]
	ax.scatter(vel, bicn, c=color, **style)
	ax.set_ylim((0.975, 1.025))
	ax.set_xlim((0, 0.2))

# %%
# reorganise gmms by weight
# dfweight = [regularise(df, sort='weights', reverse=True) for df in dflist]
# gmdisplay( dfweight[gmm4idx[0]] )


# %%
# * analyze an example double exponential-looking distribution in detail
# start by loading the PWL solution and comparing them
# index = 15
index = 4
track_idx = idx_list[index]
print('index', index, 'track_idx', track_idx)
wave_data = datalist[index]
wave_data = udatalist[index]

# load pwl
target = '/home/dan/usb_twitching/sparseml/run/cluster/no_heuristic/top/_top_1145'
solver = pwlstats.load_solver_at(target)
solver.partition.update_residuals()
curve_coord = solver.partition.get_curve_coord()
udata = np.diff(curve_coord) 
pwl_data = asymprocess(udata, q)

# %%

_xlim = (-0.05, 0.10)
with mpl.rc_context({'font.size':16}):
	fig, ax = plt.subplots(figsize=(6,4))
	sstyle = dict(element="step", fill=False, alpha=0.8, stat='density')
	sns.histplot(pwl_data, binrange=_xlim, ax=ax, linewidth=2, color=defcolor[0], **sstyle)
	sns.histplot(wave_data, binrange=_xlim, ax=ax, linewidth=2, color=defcolor[1], **sstyle)
	ax.legend(["pwl", "wavelet"])

# %%
# * also plot this PWL model
# fig, ax = plt.subplots(figsize=(200,200))
# pwlpartition.simple_model_plot(ax, solver.partition.model, data=solver.partition.data)
# ^ hard to plot because this version of PWL solver suffers rare but large errors

# %%
# * EXAMINE the positive and negative peaks for anticorrelations
# so slice the data round +/- 0.03 out of the distribution and check their time correlation?

delta = 0.01
t = 0.03
window = t-delta, t+delta
plus = np.logical_and(data > window[0], data < window[1])
window = -t-delta, -t+delta
minus = np.logical_and(data > window[0], data < window[1])

print('components {:.3f} {:.3f}'.format( data[plus].size/data.size, data[minus].size/data.size) )
# construct an array using [-1, 0, 1]
classarr = np.zeros_like(data)
classarr[plus] = 1
classarr[minus] = -1

# plt.plot(classarr[:200], linestyle='none', marker='.')
# print(list(classarr))

count_p, count_m = np.count_nonzero(plus), np.count_nonzero(minus)
corr = ( classarr[1:] * classarr[:-1] ).sum()/(count_p + count_m)
print('count plus/minus', count_p, count_m)

# in addition, explicitely count +1, -1 and -1, +1 pairs

plusidx = np.argwhere(plus).ravel()
minusidx = np.argwhere(minus).ravel()

# call these pairs forward and reverse 
f_pair = classarr[plusidx+1] == -1
r_pair = classarr[minusidx+1] == 1

nf_pair, nr_pair = np.count_nonzero(f_pair), np.count_nonzero(r_pair)
print('count forward and reverse pairs', nf_pair, nr_pair)

fraction = 2 * (nf_pair + nr_pair) / (count_p + count_m)
print('fraction of +/- in pairs', fraction)

# process data to remove +/- pairs

def remove_pairs(data, t=0.03, delta=0.003):
	data = data.copy()
	window = t-delta, t+delta
	plus = np.logical_and(data > window[0], data < window[1])
	window = -t-delta, -t+delta
	minus = np.logical_and(data > window[0], data < window[1])
	plusidx = np.argwhere(plus).ravel()
	minusidx = np.argwhere(minus).ravel()

	classdata = np.zeros_like(data)
	classdata[plus] = 1
	classdata[minus] = -1

	rmidx = np.zeros_like(data, dtype=bool)
	f_pair = classdata[plusidx+1] == -1
	r_pair = classdata[minusidx+1] == 1

	rmidx[plusidx[f_pair]] = 1
	rmidx[plusidx[f_pair]+1] = 1
	rmidx[minusidx[r_pair]] = 1
	rmidx[minusidx[r_pair]+1] = 1
	print('remove {} ({:.1f}%)'.format( np.count_nonzero(rmidx), 100*np.count_nonzero(rmidx)/rmidx.size) )

	return data[~rmidx]



mod_data = remove_pairs(data, t, delta)
print('compare size', mod_data.size, data.size)

fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(data, ax=ax, **shstyle)
sns.histplot(mod_data, ax=ax, **shstyle)
_xlim = (-0.05, 0.10)
ax.set_xlim(_xlim)
ax.axvline(t-delta, **vstyle)
ax.axvline(t+delta, **vstyle)



# plot before and after

# %%
# and try it on candidate data
# as expected, this data appears more o

index = 75
_data = datalist[75]

fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(_data, ax=ax, **shstyle)
sns.histplot(remove_pairs(_data, t, delta), ax=ax, **shstyle)
ax.set_xlim(xlim)


# %%
# * NEXT
# the next order of business is add EMG to the mixture model

