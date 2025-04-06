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
# add noise to a simulated trajectory and then investigate the instantaneous speed distribution


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

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
# pub.set_write_dir("/home/dan/usb_twitching/sparseml/EM_paper")
pub.set_write_dir("/home/dan/usb_twitching/sparseml/EM_paper/prx")
print("writing figures to", pub.writedir)


import pili
from pili import support
import twutils
import _fj
import mdl
import pwlpartition
import emanalyse

import fjanalysis
import pwlstats

import readtrack
import stats
import command
import twanalyse

# %% 
mplstyle = {"font.size": 20}
notename = "simulated_em"
plotting = True

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']


# %% 
# * Create a pulse model
# off = scipy.stats.uniform(0, 0.3)
off = scipy.stats.expon(0, 1.0)
print(off.rvs(10))
print()
on = scipy.stats.uniform(0.8, 0)
# on = scipy.stats.expon(0, 0.5)

def generate_pulse(pulse=(on, off), size=5000):
	on, off = pulse
	signal = [np.random.rand()]
	_on = on.rvs(size)
	_off = off.rvs(size)
	for i in range(size):
		signal.append(signal[-1] + _on[i])
		signal.append(signal[-1] + _off[i])
	return signal

signal = generate_pulse()
print(signal[:20])

a = np.array(signal[0::2])
b = np.array(signal[1::2])

from math import floor, ceil

size = len(b)
sample = []
for i in range(size):
	if floor(a[i]) == floor(b[i]):
		sample.append(b[i] - a[i])
	else:
		sample.append(ceil(a[i]) - a[i])
		inter = int(floor(b[i]) - ceil(a[i]))
		for _ in range(inter):
			sample.append(1.0)
		sample.append(b[i] - floor(b[i]))

sns.histplot(sample)




# %% 
rundir = join(pili.root, '../run')
path = join(rundir, "825bd8f/target/t0/")
t0 = readtrack.trackset(ddir=join(path, 'data/'))
with command.chdir(path):
	local = stats.load()
	lvel = np.load(join(path,"lvel.npy"))
	
# %% 
sns.histplot(lvel)
print("ntaut", local["ntaut"]["mean"])

# %% 

track = t0[0]

sigma = 0.009

data = track.get_head2d()

xr = np.random.multivariate_normal([0, 0], sigma**2 * np.diag([1,1]), size=len(track))

mod_data = data + xr

mtrack = track.copy()
mtrack['x'] = mod_data[:,0]
mtrack['y'] = mod_data[:,1]

# %% 

wavelet='db1'
em_config = {"wavelet":wavelet, 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}

wavemodel, lptr, meta = pwlpartition.wavelet_guess(mtrack, config=em_config)

if plotting:
	fig, ax = plt.subplots(figsize=(200,200))
	pwlpartition.simple_model_plot(ax, wavemodel, data=lptr)

# %% 

curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr, adjacent=True, ln=11)
udata = np.diff(curve_coord)

# * PREPROCESS
# q = 0.01
# data = emanalyse.asymprocess(udata, q)
# * don't cut tilas
udata[np.isnan(udata)] = 0
udata[np.isinf(udata)] = 0
data = udata[udata!=0]


# %% 
# pull simulations and plot their true velocities

# simdir = "/home/dan/usb_twitching/run/825bd8f/cluster/mc4d"
# choose_uid = np.loadtxt(join(simdir, 'standard_accepted_uid.npy'), dtype=object)

# from glob import glob
# choose_uid = glob(join(simdir, '_u_*'))
# iter_uid = itertools.count()


# # %% 
# idx = next(iter_uid)
# print(idx, choose_uid[idx])
# path = join(simdir, choose_uid[idx])

# twutils.sync_directory(path)

# _track = readtrack.trackset(ddir=join(path, 'data/'))[0]
# _origin = np.linalg.norm(_track.get_dx(), axis=1)
# _zero, _move = _origin[_origin==0], _origin[_origin!=0]
# print(_move.size, _zero.size, _origin.size)

# with mpl.rc_context({'font.size' : 30}):
# 	fig, ax = plt.subplots(figsize=(6,6))
# 	bwidth = 0.004
# 	sns.histplot(_move, ax=ax, binwidth=bwidth, **shstyle)

# 	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
# 	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
# 	ax.set_xlim(0,0.25)


# %% 

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



shstyle = dict(element="step", fill=False, alpha=0.6, lw=3)
vstyle = dict(alpha=0.3,c='black',linestyle='--')

with mpl.rc_context(usestyle):
	fig, ax = plt.subplots(figsize=fh_size)
	bwidth = 0.004

	ax.axvline(0, lw=2, **vstyle)

	sns.histplot(data, ax=ax, binwidth=bwidth, **shstyle)
	h1 = ax.lines[-1]
	_xlim = (-0.05, 0.12)
	ax.set_xlim(_xlim)


	origin = np.linalg.norm(track.get_dx(), axis=1)
	zero, move = origin[origin==0], origin[origin!=0]
	weight = move.size/origin.size
	print(origin.size, move.size, weight)

	sns.histplot(move, ax=ax, binwidth=bwidth, **shstyle)
	h2 = ax.lines[-1]

	# plot the true moving part of the distribution on top?
	# TODO cut the original data using the same indices as asymprocess
	sns.histplot(udata[origin!=0], ax=ax, binwidth=bwidth, **shstyle)
	h3 = ax.lines[-1]
	# sns.histplot(udata[origin==0], ax=ax, binwidth=0.002, **shstyle)


	normal = scipy.stats.norm(loc=0, scale=sigma).rvs(zero.size)
	# sns.histplot(normal,  ax=ax, binwidth=bwidth, **shstyle)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	_x, _y = ax.yaxis.get_label().get_position()
	ax.yaxis.set_label_coords(_x-0.24, _y)

	ax.set_xlabel(r"displacement (\textmu m)")

	# labels = [r'$X = \tilde{X} + \epsilon$']
	labels = [r'$X$'] 
	labels.append( r'$\tilde{X}|_{\tilde{X}\neq 0}$')
	labels.append( r'$X|_{\tilde{X}\neq 0}$' )
	# labels.append()
	ax.legend([h1, h2, h3], labels, handlelength=1, loc=(0.43,0.4))

pub.save_figure("simulated_noise_distributions")


# %%
# fit exponential distribution to the left tail
left = data[data<0]
scipy.stats.anderson(-left, dist='expon')

d = udata[origin==0]
z = d[d!=0]
zleft = z[z<0]
# np.isnan(zleft).sum()

scipy.stats.anderson(-zleft, dist='expon')


# %%
# * LOAD
# check if this exponential behaviour is a consequence of the measurement window
# we can do this by counting the number of effective retraction events per interval

tdir = join(pili.root, '../run/', '825bd8f/target/t0/detail/')
detail_track = readtrack.Track( join(tdir, 'trackxy.dat') )
detail_mdev = readtrack.Track( join(tdir, 'mdev.dat') )

# %%
# displacement per retraction event

def get_dx(md):
	md = md.copy()
	md.filter_by_process("retraction")
	dx, dy = md['d_x'], md['d_y']
	d = np.column_stack([dx, dy])
	return np.linalg.norm(d, axis=1)

# * PLOT 
# the displacements associated with simulated retraction events of 4 nm
udx = get_dx(detail_mdev)
dx = udx[udx != 0]
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(dx, ax=ax, **shstyle)
ax.axvline(0.004, **vstyle)
ax.set_xlim((0, 0.015))

# %%
# TODO study single retraction events
# TODO check the displacement per pilus value

# %%
# * COUNT retraction events in 0.1s intervals 
# counting retraction events

md = detail_mdev.copy()
md.filter_by_process("retraction")
time = md['time']
interval_idx = np.searchsorted(time, np.arange(0, 2000, 0.1))
count_rt = np.diff(interval_idx)

zero, retc = count_rt[count_rt == 0], count_rt[count_rt != 0]
print('count intervals with zero/some bound-retraction events')
print( zero.size, retc.size, zero.size/20000)

def plot_retc(ax, retc, fontsize=None):
	sns.histplot(retc, binwidth=1, stat='density', ax=ax, color=defcolor[0], **shstyle)
	print(scipy.stats.anderson(retc, dist='expon'))

	ax.set_xlim(0, 30)
	ax.set_ylabel('Density')
	ax.set_xlabel('No. retractions')
	# fit
	xspace = np.linspace(1, 40, 1000)
	loc, scale = scipy.stats.expon.fit(retc)
	print('1/scale', 1/scale)
	pdf = scipy.stats.expon(loc, scale).pdf(xspace)
	ax.plot(xspace, pdf, lw=3, alpha=0.8, color=defcolor[1], linestyle='--')

	ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	# ax.yaxis.set_ticks([])
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))

	# ax.legend(['simulation', r'$f(x; \lambda = 0.24)$'])
	ax.legend(['simulation', r'$\lambda = 0.24$'], handlelength=1)
	
with mpl.rc_context({'font.size' : 30}):
	fig, ax = plt.subplots(figsize=(6,6))

	plot_retc(ax, retc)



# %%
# fit exponential part and plot it
# true nonzero dx

loc, scale = scipy.stats.expon.fit(move)
lam = 1/scale
print('lambda', lam)

xspace = np.linspace(0, 0.15, 1000)
pdf = scipy.stats.expon(0, scale).pdf(xspace)

with mpl.rc_context(texstyle):
	# fig, axes = plt.subplots(2, 1, figsize=use_size)
	fig, axes = plt.subplots(2, 1, figsize=use_size)
	ax = axes[0]

	sns.histplot(move, ax=ax, binwidth=bwidth, stat='density', color=defcolor[0], **shstyle)
	ax.plot(xspace, pdf, lw=4, alpha=0.8, color=defcolor[1], linestyle='--')

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(2))

	ax.set_xlim(-0.01, _xlim[1])
	ax.set_xlabel("displacement $(\mu m)$")

	# ax.legend([r'$\Delta s|_{\Delta s\neq 0}$', r'$\mathrm{f}(x; \lambda = 40.1)$'])
	ax.legend([r'$\tilde{X}|_{\tilde{X}\neq 0}$', r'$\lambda = 40.1$'])
	
	ax = axes[1]
	plot_retc(ax, retc)

	# inset = ax.inset_axes([0.72, 0.24, 0.45, 0.45])
	# plot_retc(inset, retc, fontsize=20)

	# inset the plot that counts retraction events


# %%

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(1, 1, figsize=use_size)

	# sns.histplot(move, ax=ax, binwidth=bwidth, stat='density', color=defcolor[0], **shstyle)
	# ax.plot(xspace, pdf, lw=4, alpha=0.8, color=defcolor[1], linestyle='--')

	# ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	# ax.yaxis.set_major_locator(plt.MaxNLocator(2))

	# ax.set_xlim(-0.01, _xlim[1])
	# ax.set_xlabel("displacement $(\mu m)$")

	# # ax.legend([r'$\Delta s|_{\Delta s\neq 0}$', r'$\mathrm{f}(x; \lambda = 40.1)$'])
	# ax.legend([r'$\tilde{X}|_{\tilde{X}\neq 0}$', r'$\lambda = 40.1$'])
	
	plot_retc(ax, retc)
	from matplotlib.ticker import ScalarFormatter

	sf = ScalarFormatter(useOffset=False)
	sf.set_powerlimits((1,-1))
	ax.yaxis.set_major_formatter( sf )
	ax.yaxis.set_ticks([0, 0.1, 0.2])
	# ax.xaxis.set_major_locator(plt.MaxNLocator(3))


pub.save_figure("simulated_displacement_distribution")

# %%
scipy.stats.anderson(move, dist='expon')


# %%
# * EXAMINE a simulated walking bacteria

# tdir = join(pili.root, '../run/', '825bd8f/target/t0/detail/')
# walk_track  = readtrack.Track( join(tdir, 'trackxy.dat') )
# walk_mdev = readtrack.Track( join(tdir, 'mdev.dat') )

path = join(rundir, "825bd8f/target/t2/")
t2 = readtrack.trackset(ddir=join(path, 'data/'))
# with command.chdir(path):
# 	local = stats.load()
# 	lvel = np.load(join(path,"lvel.npy"))

# %%

track = t2[0]
sigma = 0.009

data = track.get_head2d()

xr = np.random.multivariate_normal([0, 0], sigma**2 * np.diag([1,1]), size=len(track))
mod_data = data + xr

mtrack = track.copy()
mtrack['x'] = mod_data[:,0]
mtrack['y'] = mod_data[:,1]

# %% 

wavemodel, lptr, meta = pwlpartition.wavelet_guess(mtrack, config=em_config)

if plotting:
	fig, ax = plt.subplots(figsize=(200,200))
	pwlpartition.simple_model_plot(ax, wavemodel, data=lptr)


# %% 
origin = np.linalg.norm(track.get_dx(), axis=1)

curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr)
udata = np.diff(curve_coord)

# * PREPROCESS
# q = 0.01
# data = emanalyse.asymprocess(udata, q)
# * don't cut tilas
# udata[np.isnan(udata)] = 0
# udata[np.isinf(udata)] = 0
ud = udata[origin!=0]

origin_nonzero = origin[origin!=0]
print('origin size', origin_nonzero.size, origin.size)

print( (ud==0).sum(), ud.size, 100*(ud==0).sum()/ud.size )

data = udata[udata!=0]
# %% 


shstyle = dict(element="step", fill=False, alpha=0.6, lw=5)
vstyle = dict(alpha=0.3,c='black',linestyle='--')

with mpl.rc_context({'font.size' : 30}):
	fig, ax = plt.subplots(figsize=(6,6))
	bwidth = 0.004

	ax.axvline(0, lw=3, **vstyle)

	sns.histplot(data, ax=ax, binwidth=bwidth, **shstyle)
	h1 = ax.lines[-1]
	_xlim = (-0.05, 0.20)
	ax.set_xlim(_xlim)


	zero, move = origin[origin==0], origin[origin!=0]
	weight = move.size/origin.size
	print(origin.size, move.size, weight)

	# plot the true non-zero part
	print(scipy.stats.anderson(move, dist='expon'))

	sns.histplot(move, ax=ax, binwidth=bwidth, **shstyle)
	h2 = ax.lines[-1]

	# plot the corrupted non-zero part
	# TODO cut the original data using the same indices as asymprocess
	ud = udata[origin!=0]
	corrupted_nonzero = ud[ud!=0]
	# sns.histplot(ud, ax=ax, binwidth=bwidth, **shstyle)
	sns.histplot(corrupted_nonzero, ax=ax, binwidth=bwidth, **shstyle)
	h3 = ax.lines[-1]


	normal = scipy.stats.norm(loc=0, scale=sigma).rvs(zero.size)
	# sns.histplot(normal,  ax=ax, binwidth=bwidth, **shstyle)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	ax.set_xlabel("displacement ($\mu m$)")

	labels =['corrupted $\Delta s$']
	labels.append( r'$\Delta s|_{\Delta s\neq 0}$')
	labels.append( r'corrupted $\Delta s|_{\Delta s\neq 0}$' )
	# labels.append()
	ax.legend([h1, h2, h3], labels, fontsize=16)

# %% 


# %% 
# attempt to preprocess this simulated data using pwltree
import pwltree

def pwl_tree_solve(data, r):
	tsolver = pwltree.TreeSolver(data, overlap=True)
	# tsolver.build_initial_tree(wavemodel)
	tsolver.build_max_tree()
	tsolver.build_priority()
	tsolver.solve(pwltree.stop_at(r))
	return tsolver

def get_lptr(track):
	dt = np.insert(np.diff(track['time']), 0, 0)
	lptr = mdl.LPtrack(dt, track['x'], track['y'])
	return lptr


r_stop = pwlpartition.estimate_r(sigma)
print('r', r_stop)
track_data = get_lptr(mtrack) #! use mtrack
short_data = track_data.cut(0,200)

if False:
	with support.Timer():
		tree_solver = pwl_tree_solve(short_data, r_stop)

# %% 
tree_model = tree_solver.get_model()

if plotting:
	fig, ax = plt.subplots(figsize=(20,20))
	pwlpartition.simple_model_plot(ax, tree_model, data=short_data)

# %% 
# compute the fraction of zeros again
origin = np.linalg.norm(track.cut(0,2002).get_dx(), axis=1)
origin_nonzero = origin[origin!=0]
print('origin size', origin_nonzero.size, origin.size)

curve_coord = emanalyse.compute_curve_coordinate(tree_model, short_data, adjacent=False)
udata = np.diff(curve_coord)
ud = udata[origin!=0]
print( (ud==0).sum(), ud.size, 100*(ud==0).sum()/ud.size )


