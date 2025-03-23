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
# sample entropy for biological trajectories

# %%
import time
import sys, os
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import multiprocessing as mp

from sctml import information
from sctml import publication as pub
sampen =  information.sampen
Timer = information.Timer

import pili
from pili import support
import twanalyse
import readtrack
import stats
import _fj
import command

# %%
import thesis
import thesis.publication as thesis

# %%
# construct 1d persistent random walk data
N = int(1e4)
print("using sequences of length ", N)
# expensive
# N = int(1e5)
#
def generate_prw(q, a, N):
	av = np.random.normal(size=N)
	v = [1]
	for i in range(N):
		v.append(q*v[i] + a*av[i])
	return np.array(v)
# %%
q = 1
a = 0.1
prwdata = generate_prw(q, a, N)
prwx = np.cumsum(prwdata)

# %%
def plot_prw(ax, prwx):
	N = len(prwx)
	_disp =  np.cumsum(np.full(N, 0.1))
	l, = ax.plot(prwx, _disp)
	return l
a_lspace = [0.1, 0.2, 0.5, 1.0]
q = 0.8
fig, ax = plt.subplots()
artists = []
for avar in a_lspace:
	prwv = generate_prw(q, avar, N)
	prwx = np.cumsum(prwv)
	l = plot_prw(ax, prwx)
	artists.append(l)
ax.legend(artists, a_lspace)

# %%
# explore sample entropy hyperparameters
def regularise(X):
	return (X - np.mean(X))/np.std(X)
prwv = generate_prw(0.8, 0.1, N)
data = regularise(prwv)
m = 2
rf = 0.1
r = 0.1 * np.std(prwv)
with Timer("sampen") as t:
	sample_entropy = sampen(data, m, r)

# %%
if False:
	rfbasis = [0.1,0.2,0.3,0.4]
	datastd = np.std(data)
	r_entropy = []
	for rf in rfbasis: 
		r = rf*datastd
		entropy = sampen(data, m, r)
		r_entropy.append(entropy)
# %%
if False:
	plt.plot(rfbasis, r_entropy, marker='o')

# %%
# multiscale entropy analysis
# https://pubmed.ncbi.nlm.nih.gov/12190613/
def granular(data, n):
	N = len(data)
	nt = N // n
	data = data[:n*nt] 
	# ^ split must be exact
	return np.array([np.mean(arr) for arr in np.split(data, nt)])

def resample(data, n):
	return data[::n]


def multiscale_sampen(data, m, r, timescale=None, granular=granular):
	if timescale is None:
		max_timescale = 10
		timescale = list(range(1,max_timescale+1))

	tau_entropy = []
	for tau in timescale:
		coarse_data = granular(data, tau)
		entropy = sampen(coarse_data, m, r)
		tau_entropy.append(entropy)
	return np.array(timescale), np.array(tau_entropy)

r = 0.2 * np.std(data)
m = 2
# (q,a)
# timescale, tau_entropy = multiscale_sampen(data, m, r)

# %%
# quick plotting that multiscale entropy that we have so far
if False:
	ax = plt.gca()
	l, = ax.plot(timescale, tau_entropy)
	artists = [l]
	labels = [(0.8,0.1)]
	ax.legend(artists, labels)
# %%
# generalise to random walks of several parameters
parlist = [(0.0,1.0),(0.0,0.5),(0.8,0.1),(0.8,0.3),(0.8,0.5),(0.2, 0.1),(0.2, 0.5),(1.0,0)]
m = 2
rf = 0.2
# generate data
pardata = []
for (q,a) in parlist:
	prwv = generate_prw(q,a, N)
	pardata.append(prwv)
# standard deviatin over ALL samples 
# we need to use the same r otherwise entropy is not comparable
r = rf * np.std(np.concatenate(pardata))
pool = mp.Pool(8)
arglist = [(data, m, r) for data in pardata]

with Timer("sampen pool") as t:
	result = pool.starmap(multiscale_sampen, arglist)

# 1 processor
# itertools.starmap(multiscale_sampen, arglist)

# %%

def plot_result(ax, result, parlist):
	artists = []
	for res in result:
		basis, entropy = res
		l, = ax.plot(basis, entropy)
		artists.append(l)
	labels = ['q={},a={}'.format(q,a) for q,a in parlist]
	ax.legend(artists, labels, bbox_to_anchor=(1.04,1))



fig, ax = plt.subplots(figsize=(7,5))
dt = 0.1
_result = [(dt*basis, entropy) for basis, entropy in result]
plot_result(ax, _result, parlist)
ax.set_ylabel("sample entropy")
ax.set_xlabel("timescale (s)")
fig.tight_layout()

pub.save_figure("prw_entropy", __file__)

# %%
# what about the velocity of persistence random walk in 2d

def generate_prw2d(q, a, N):
	mean = [0,0]
	cov = np.identity(2)
	av = np.random.multivariate_normal(mean, cov, size=N)
	v = [np.array((0, 1))]
	for i in range(N):
		v.append(q*v[i] + a*av[i])
	return np.array(v)

prw2v = generate_prw2d(0.8, 0.1, N)

def plot_prw2(ax, vdata):
	N = len(prwx)
	xdata = np.cumsum(vdata,axis=0)
	l, = ax.plot(xdata[:,0], xdata[:,1])
	return l
plot_prw2(plt.gca(), prw2v)

# %%
q = 0.8
a = 0.1
prw2v = generate_prw2d(q, a, N)
norm = np.linalg.norm
vdata = norm(prw2v, axis=1)
rf = 0.2
r = rf * np.std(vdata)

# %%
# generate data
pardata = []
for (q,a) in parlist:
	prw2v = generate_prw2d(q,a, N)
	prwv = norm(prw2v, axis=1)
	pardata.append(prwv)
# standard deviatin over ALL samples 
# we need to use the same r otherwise entropy is not comparable
r = rf * np.std(np.concatenate(pardata))
pool = mp.Pool(8)
arglist = [(data, m, r) for data in pardata]

with Timer("sampen pool") as t:
	result = pool.starmap(multiscale_sampen, arglist)

ax = plt.gca()
plot_result(ax, result, parlist)

# %%
# load tmos data
simdir = join(pili.root, "../run/825bd8f/target/t0")
# load summary statistics
with command.chdir(simdir):
	t0 = readtrack.trackset()
	ld = stats.load()

# use head velocity, not step velocity
qa = twanalyse.mle_persistence(t0)
t0q, t0a  = qa['qhat']['estimate'], qa['ahat']['estimate']
print('tmos (q,a) = ', t0q, t0a)

# %%
# load linearised tracking data
subsets = _fj.load_subsets()
top_idx = _fj.load_subset_idx()["top"]
lintop = subsets['top']

# load denoised tracking data
top = _fj.trackload(top_idx)

# load original tracking data
original_top = _fj.trackload_original(top_idx)


# %%
# head velocity
def get_vel(trs):
	return np.concatenate([tr.get_head_v() for tr in trs])

tmos_vel = get_vel(t0)
speed = norm(tmos_vel, axis=1)
tmos_vdata = speed[:N]
print('keeping {:.1%} of simulated data'.format(N/speed.size))

track_vel = get_vel(top)
speed = norm(track_vel, axis=1)
track_vdata = speed[:N]
print('keeping {:.1%} of fanjin tracking data'.format(N/speed.size))
#

# lin_tracking_velocity  = np.concatenate([tr.get_head_v() for tr in lintop])
# lin_tracking_speed = norm(lin_tracking_velocity, axis=1)
# lin_track_vdata = lin_tracking_speed[:N]
# #

original_track_vel = get_vel(original_top)
speed = norm(original_track_vel, axis=1)
original_track_vdata = speed[:N]

# %%
# basis, entropy = multiscale_sampen(tmos_vdata, m, r)
# fig, ax = plt.subplots(figsize=(5,5))
# ax.plot(basis,  entropy)

# %% [markdown]
# tmos data shows exactly the effect we want, the sample entropy takes a sharp dive at low coarse graining

# %% [markdown]
# for this application linearising our data using a length creates complications
# - for small dstep, large jumps will not be resolved in 0.1s output
# - If we use shorter timescale output we need to re-run out simulations, genearate lots of data
#   and then be carefull that sampen function is fast enough for the shortest dstep
# see linsampen.py


# %% 
N = int(1e4)
N
# %% 
# %% 
import scipy.fft

def pink_noise(f):
	return 1/np.where(f == 0, float('inf'), np.sqrt(f))

white = np.fft.rfft(np.random.randn(N))
S = pink_noise(np.fft.rfftfreq(N))
S = S/np.sqrt(np.mean(S**2))
X = white * S
pink = np.fft.irfft(X)
print(pink.size)

# %% 

white = np.fft.rfft(np.random.rand(N))
S = np.fft.rfftfreq(N)
S = S/np.sqrt(np.mean(S**2))
X = white * S
white = np.fft.irfft(X)


# %% 

def regularise(X, std_value=None):
	if std_value is None:
		std_value = np.std(X)
	return (X - np.mean(X))/std_value

# %% 
lstyle = dict(alpha=0.8, lw=2.5)
def coarsen_plot(ax, data, n=300, cf = granular):
	scheme = iter(["#052CCC", "#058FCC", '#05CC42'])
	displace = 1.5 * np.quantile(data, 0.99)
	ax.plot(data[:n], color=next(scheme), **lstyle)
	ax.plot(cf(data[:3*n], 3)-displace, color=next(scheme),**lstyle)
	ax.plot(cf(data[:15*n], 15)-2*displace, color=next(scheme), **lstyle)

	ax.set_xlabel("index")
	# ax.set_yticks([])
	# ax.set_ylabel(r'timescale $\tau$ (s)')
	# ax.set_yticks([0, -displace, -2*displace], labels=[0.1, 0.3, 1.5])
	ax.set_yticks([0, -displace, -2*displace])
	ax.set_yticklabels([])


fsize = (6,4)
title_fsize = 26

with mpl.rc_context(thesis.texstyle):
	fig, ax  = plt.subplots(figsize=fsize)
	coarsen_plot(ax, pink)
	ax.set_title(r"$1/f$")
	# ax.set_yticks([])
	# ax.set_ylabel('velocity signal')

	scheme = iter(["#052CCC", "#058FCC", '#05CC42'])
	from matplotlib.lines import Line2D
	llstyle = lstyle.copy()
	llstyle['lw'] = 8
	h1 = Line2D([0], [0], color=next(scheme), **llstyle)
	h2 = Line2D([0], [0], color=next(scheme), **llstyle)
	h3 = Line2D([0], [0], color=next(scheme), **llstyle)
	ax.legend([h1, h2, h3], ['0.1 s', '0.3 s', '1.5 s'], title=r"timescale, $\tau$", bbox_to_anchor=(0.75,-0.3))

thesis.save_figure('pink_cg_signal')

# %%

std_value = np.mean([np.std(tmos_vdata), np.std(original_track_vdata), np.std(track_vdata)])
simv = regularise(tmos_vdata, std_value)
trackv = regularise(original_track_vdata, std_value)
smooth_trackv = regularise(track_vdata, std_value)

with mpl.rc_context(thesis.texstyle):
	fig, ax  = plt.subplots(figsize=fsize)
	coarsen_plot(ax, simv)
	ax.set_title(r"simulation")
	ax.set_xlabel('')
	ax.set_xticklabels([])
thesis.save_figure('sim_cg_signal')



with mpl.rc_context(thesis.texstyle):
	fig, ax  = plt.subplots(figsize=fsize)
	coarsen_plot(ax, trackv)
	ax.set_title(r"tracking")
	ax.set_xlabel('')
	ax.set_xticklabels([])
	
	ax.set_ylabel('')
	ax.set_yticklabels([])
thesis.save_figure('tracking_cg_signal')
print(ax.get_ylim())


with mpl.rc_context(thesis.texstyle):
	fig, ax  = plt.subplots(figsize=fsize)
	coarsen_plot(ax, smooth_trackv)
	ax.set_title(r"denoised tracking")

	ax.set_ylabel('')
	ax.set_yticklabels([])
	plt.draw()
	ax.set_ylim(ax.get_ylim()[0] ,4.5285287996437384)

thesis.save_figure('denoised_cg_signal')


gen_vel= generate_prw2d(t0q, t0a, N)
gen_vdata = norm(gen_vel, axis=1)

with mpl.rc_context(thesis.texstyle):
	fig, ax  = plt.subplots(figsize=fsize)
	coarsen_plot(ax, regularise(gen_vdata)[10:])
	ax.set_title(r"random walk")
thesis.save_figure('randomwalk_cg_signal')

# %% 
# check standard deviations
datasets = []
datasets.append(tmos_vdata)
datasets.append(gen_vdata)
datasets.append(original_track_vdata)
datasets.append(track_vdata)
[np.std(v) for v in datasets]





# %% 

n = 600
r = 0.15 * np.std(pink)
print('r', r)

cf = granular
print(sampen(cf(pink[:n], 1), m, r))
print(sampen(cf(pink[:n], 2), m, r))
print(sampen(cf(pink[:n], 5), m, r))
print(sampen(cf(pink[:n], 10), m, r))
print(sampen(cf(pink[:n], 20), m, r))


# %% 
timescale = range(1,20,1)
with support.Timer():
	white = np.random.rand(N)
	r = 0.2 * np.std(white)
	res = multiscale_sampen(white, m, r, timescale=timescale)
	plt.plot(*res)
basis, white_sampen = res

# %% 
white_sampen

# %% 
with support.Timer():
	r = 0.2 * np.std(pink)
	res = multiscale_sampen(pink, m, r, timescale=timescale)
	plt.plot(*res)

basis, pink_sampen = res

# %% 
# Next: we want to compare persistent random walk data with tmos data
# its recommended to regularise trajectories using (X - np.mean(x))/np.std(X) before comparison
# we also choose r based on the standard deviation, so choose that afterwards
timescale = range(1,20,1)
rf = 0.2
m = 2

# construct datasets in advance in case we want to choose r
vdatasets = []
vdatasets.append(regularise(tmos_vdata))
vdatasets.append(regularise(gen_vdata))
vdatasets.append(regularise(original_track_vdata))
vdatasets.append(regularise(track_vdata))


# %%

msampen = []
r = 0.2 * np.std(np.concatenate(vdatasets))
# simulated twitching
with support.Timer():
	for vdata in vdatasets:
		res = multiscale_sampen(vdata, m, r, timescale=timescale)
		msampen.append(res)


# %% 

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
def generate_linestyles(color=defcolor, markersize=8):
	c = iter(color)
	marker = ['.', 'o', 'x']

	# dict(color=color[1], marker='o', markeredgecolor=color[1], markerfacecolor='none'),

	linestyles = [
		dict(color=color[0], marker='o', markerfacecolor='none'),
		dict(color=color[1], marker='x'),
		dict(color=color[2], marker='s', markerfacecolor='none'),
		dict(color=color[3], marker='v', markerfacecolor='none'),
	]
	for l in linestyles:
		l['markersize'] = markersize
		l['markeredgewidth'] = 1.5
	return linestyles


mycolor = defcolor
del mycolor[1]
mycolor[3] = mycolor[2]
linestyles = generate_linestyles(mycolor, markersize=10)


# %% 

def formatqa(q,a):
	return "q={:.1f},a={:.1f}".format(q,a)
labels = []
labels.append("simulation")
labels.append(formatqa(t0q,t0a))
labels.append("experiment")
labels.append("wavelet")

from itertools import cycle
lstyle = cycle(linestyles)
with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(5,5))
	artists = []
	dt = 0.1
	for (basis, entropy) in msampen:
		l, = ax.plot(dt*basis, entropy, alpha=0.7, **next(lstyle))
		artists.append(l)

	# pink_color = '#9505CC'
	pink_color = defcolor[5]
	pink_style = dict(linestyle='none', marker='s', markerfacecolor='none', markersize=10, markeredgewidth=1.5, c=pink_color)
	ideal_line = dict(linestyle='-', linewidth='4', color=pink_style['c'])

	ax.plot(dt*basis, np.full(basis.size, np.mean(pink_sampen)), alpha=0.3, zorder=-1, **ideal_line)
	h, = ax.plot(dt*basis, pink_sampen, **pink_style)
	labels.append(r'$1/f$ noise')
	artists.append(h)

	white = False
	if white:
		white_color = defcolor[-3]
		white_style = dict(linestyle='none', marker='s', markerfacecolor='none', markersize=8, c=white_color)
		# ax.plot(dt*basis, white_sampen, alpha=1.0, zorder=-1, marker='o', markerfacecolor='none', marker)
		ideal_line = dict(linestyle='-', linewidth='6', color=white_style['c'])
		x, y = dt*basis, white_sampen,
		z = np.polyfit(x, y, 4)
		p = np.poly1d(z)
		h, = ax.plot(dt*basis, p(dt*basis), alpha=0.3, zorder=-2, **ideal_line)
		labels.append(r'white noise')
		artists.append(h)


	# ax.legend(artists, labels, loc=(1.0,0.0))
	ax.set_ylabel("sample entropy")
	ax.set_xlabel("timescale (s)")


	ax.set_xlim(0,2)
	ax.xaxis.set_major_locator(plt.MaxNLocator(5))
	ax.yaxis.set_major_locator(plt.MaxNLocator(5))

	thesis.bold_ticks(ax)

	# pub.save_figure("sample_entropy", __file__)

	thesis.save_figure("sample_entropy", __file__)

#  but are we comparing like with like?
# we should compute q,a for the 0.1s resolution simulated data, right?

# %% 

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(5,5))
	ax.legend(artists, labels, loc='center')
	ax.axis(False)
	thesis.save_figure("sample_entropy_legend", __file__)
	

# %% 
def safe_norm(v):
	# expected v.shape  = (N, 2)
	n = norm(v,axis=1)
	mask = n > 0
	v[mask] = v[mask]/n[mask][:,np.newaxis]
	return v

def correlate2d(v1, v2, max_tau=20):
	# v1 = v1/norm(v1,axis=1)[:,np.newaxis]
	# v2 = v2/norm(v2,axis=1)[:,np.newaxis]
	v1 = safe_norm(v1)
	v2 = safe_norm(v2)
	def auto(tau):
		return np.mean(np.sum(v1[tau:]*v2[:-tau], axis=1))
	return np.array(list(map(auto, range(1,max_tau+1))))

# %%
# collect the 2d datasets
v2datasets = [tmos_vel, gen_vel, original_track_vel, track_vel]
	
# %% 


# %% 
# ^same but for 2d autocorrelation

lstyle = iter(linestyles)
with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(5,5))
	artists = []
	l2style = {"alpha":1.0, "linestyle":'--', "linewidth":2.0}
	for v2data in v2datasets:
		acorr = correlate2d(v2data, v2data, max_tau=mtau)
		l, = ax.plot(basis, acorr, **next(lstyle))
		artists.append(l)

	ax.set_ylim((-0.3,None))
	ax.set_xlim(-0.1,2.0)
	ax.set_ylabel(r'$\langle v(t) \cdot v(t + \tau)\rangle$')
	ax.set_xlabel('timescale (s)')
	# ax.legend(artists, labels)

	ax.xaxis.set_major_locator(plt.MaxNLocator(5))
	ax.yaxis.set_major_locator(plt.MaxNLocator(5))

	thesis.bold_ticks(ax)
	thesis.save_figure('velocity_autocorrelation')
 = ax

# %%

lstyle = iter(linestyles)
with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(5,5))
	s = N//2  + 1
	basis = np.arange(0, 2, 0.1)
	mtau = 20
	artists = []
	for vdata in vdatasets:
		autocorr = np.correlate(vdata, vdata, mode="same")
		l, = ax.plot(basis, autocorr[s:s+mtau]/N, **next(lstyle))
		artists.append(l)

	ax.set_ylim(prev_ax.get_ylim())
	# ax.set_yticks(np.arange(-0.3,0.91,0.3))
	ax.set_xlim(-0.1,2.0)
	ax.set_ylabel(r'$\mathrm{corr}(v,v)$')
	ax.set_xlabel('timescale (s)')
	# ax.legend(artists, labels)

	ax.xaxis.set_major_locator(plt.MaxNLocator(5))
	# ax.yaxis.set_major_locator(plt.MaxNLocator(5))
	thesis.bold_ticks(ax)

	thesis.save_figure('speed_correlation')

