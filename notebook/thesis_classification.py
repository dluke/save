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
# redesign our classification procedures with both PWL and simulation work
# straight forward analysis on this bacteria population for thesis

# we start by getting the duration, end-to-end displacement, no. linear steps, etc.

# %% 
import os
import sys
import json
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
norm = np.linalg.norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import shapeplot

import _fj
import plotutils
import command
import twanalyse
import fjanalysis
import twanimation
import pili
import stats

import pandas as pd
import seaborn as sns
import pili.publication as pub

sys.path.append('/home/dan/usb_twitching/')
import thesis.publication as thesis

import pili.support as support
from pili.support import make_get, get_array

from tqdm import tqdm


# %% 
verbose = False
animate = False
work = False

# %% 
debug = None
idx, original_trs = _fj.slicehelper.load_original_trs('all', debug)
idx, trs = _fj.slicehelper.load_trs('all', debug)
idx, ltrs = _fj.slicehelper.load_linearized_trs('all', debug)


# %% 
def load_outlier_data():
	with command.chdir(join(pili.root, "src/analysis/")):
		local = stats.load()
	return local

local = load_outlier_data()

# %%
# load experimental data statistics
datalist = fjanalysis.load_summary()
qhat = get_array(make_get('qhat.estimate'), datalist)
ahat = get_array(make_get('ahat.estimate'), datalist)

# %%

def displacement_length(tr): # end-to-end displacement length
	x, y = tr['x'], tr['y']
	return np.sqrt( (x[0]-x[-1])**2 + (y[0]-y[-1])**2 )

def min_aspect(tr):
	def _coarse_graining(arr, framewindow=200):
		# print('Uniform coarse graining with width = {}'.format(framewindow))
		if arr.size <= framewindow:
			return np.nan
		frh = framewindow//2
		sbasis = np.arange(frh, arr.size-frh, 1, dtype=int)
		return np.array([np.mean(arr[j-frh:j+frh-1]) for j in sbasis])
	min_lwratio = np.array([np.min(_coarse_graining(tr['length']/tr['width'])) for tr in tqdm(trs)])

def _tangent(dx, max_distance=2.0):
	# support = 0.12 * np.arange(1,max_index)
	support = np.arange(0.12, max_distance, 0.12)
	ndx = dx/norm(dx,axis=1)[:,np.newaxis]
	n = norm(dx,axis=1)[:,np.newaxis]
	corr = []
	for shift in range(1, len(support)+1):
		val = np.nanmean( (ndx[shift:]*ndx[:-shift]).sum(axis=1) )
		corr.append(val)
	return support, np.array(corr)

# compute min of tangent correlation for all trajectories
tangent_data = [_tangent(ltr.get_step_dx())[1] for ltr in ltrs]
min_tangent_corr = np.array([min(arr) for arr in tangent_data])
first_tangent_corr = np.array([arr[0] for arr in tangent_data])


data = {
	"duration" : [tr.get_duration() for tr in trs],
	"displacement" : [displacement_length(tr) for tr in trs],
	# "min_aspect" : [min_aspect(tr) for tr in trs],
	"min_aspect" : local["min_aspect_ratio"]
}

ldata = {
	"nsteps" : [ltr.get_nsteps() for ltr in ltrs],
	"step_displacement" : [np.sum(norm(ltr.get_step_dx(),axis=1)) for ltr in ltrs],
	"lvel" : [twanalyse.lvel(ltr) for ltr in ltrs],
	"qhat" : qhat,
	"ahat" : ahat,
	"min_tangent_corr" : min_tangent_corr,
	"first_tangent_corr" : first_tangent_corr
}

df = pd.DataFrame({**data, **ldata})
df["meanvel"] = df["displacement"]/df["duration"] # end to end displacement per time (why?)
df

# %%
df.sort_values("meanvel")

# %%
# cache
# actually save later
df.to_pickle("classification.pkl") 

# %%
# * PAPER FIGURES

threshold = 1.6

# texstyle = thesis.texstyle.copy()
# plt.rc('text.latex',  preamble=[r'\usepackage{sfmath} \boldmath'])
# texstyle["font.size"] = 28

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(6,4))

	hstyle = {'stat':'count', 'alpha':0.4, 'element':'step'}
	data = df["min_aspect"]

	# the bins we will use
	entries, bins = np.histogram(data, bins='auto')
	bins = np.arange(1.0,6.6,0.2)

	left_data = data[threshold > data]
	right_data = data[threshold < data]

	sns.histplot(left_data, bins=bins[:5], **hstyle)
	sns.histplot(right_data, bins=bins[3:], color='salmon', **hstyle)
	ax.axvline(1.6, linestyle='--', color='k', alpha=0.3)
	ax.set_xlabel(r"$b_{\mathrm{min}}$", fontsize=24)
	ax.legend(["walking", "crawling"])

	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	thesis.bold_ticks(ax)


# plot_target = join(pili.root, "../writing/draw")

# 
pub.save_figure("aspect_min_distribution", "paper")
thesis.save_figure("aspect_min_distribution")


# %%
data = df["displacement"]/df["step_displacement"]
# TODO
# a few values are > 1, which is impossible, I checked them and they have 
# nearly zero steps

with mpl.rc_context({"font.size" : 20, "text.usetex": True}):
	fig, ax = plt.subplots(figsize=(6,4))
	xlim = (0,1)
	sns.histplot(data, binrange=xlim, ax=ax)
	ax.set_xlabel(r"$h/L_s$", fontsize=24)


# since this distrubtion is also biomodel, its interesting to see if it correlates with horizontal/vertical orientation

# %%
# data = df["displacement"]
# h1 = data[df["min_aspect"] < 1.6]
# h2 = data[df["min_aspect"] > 1.6]

# sns.histplot( h2 )
# sns.histplot( h1 )

# %%
# * there is a correlation between these two peaks and walking/crawling
data = df["displacement"]/df["step_displacement"]
h1 = data[df["min_aspect"] < 1.6]
h2 = data[df["min_aspect"] > 1.6]

ylim = (0,320)

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	ax.set_ylim(ylim)
	hstyle = hstyle.copy()
	xlim = (0,1)
	hstyle.update( dict(binrange=xlim) )
	sns.histplot(h2, ax=ax, color="salmon", **hstyle)
	sns.histplot(h1, ax=ax, **hstyle)
	ax.legend(["crawling", "walking"], fontsize=16)
	ax.set_xlabel(r"$h/C$", fontsize=24)

	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	thesis.bold_ticks(ax)

thesis.save_figure("h_population_distribution")



# %%
# same but use the displacement threshold
data = df["displacement"]/df["step_displacement"]

h1 = data[df["displacement"] < 3.0]
h2 = data[df["displacement"] > 3.0]

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	ax.set_ylim(ylim)
	hstyle = hstyle.copy()
	hstyle.update( dict(binrange=xlim) )
	sns.histplot(h2, ax=ax, color="salmon", **hstyle)
	sns.histplot(h1, ax=ax, **hstyle)
	ax.legend(["$h > 3.0$", "$h < 3.0$"], fontsize=16)
	ax.set_xlabel(r"$h/C$", fontsize=24)

	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	thesis.bold_ticks(ax)


thesis.save_figure("h_population_distribution_threshold")


# %%
# hc = data[crawling]
# hw = data[walking]

# fig, ax = plt.subplots(figsize=(6,4))
# sns.histplot(hc, ax=ax)
# sns.histplot(hw, ax=ax)


# %%
# plot track 288
interest = ltrs[288]

fig, ax = plt.subplots(figsize=(6,4))
x, y = interest.get_step_head().T
plt.plot(x,y, marker='o')

# shapeplot.longtracks(ax, [interest])


# %%
# load subset data
subset = _fj.load_subsets()
wdatalist = [twanalyse.observables([ltr]) for ltr in subset["walking"]]
wvel = [wd["lvel"]["mean"] for wd in wdatalist]

cdatalist = [twanalyse.observables([ltr]) for ltr in subset["top"]]
cvel = [cd["lvel"]["mean"] for cd in cdatalist]

with mpl.rc_context({"font.size" : 20, "text.usetex": True}):
	fig, ax = plt.subplots(figsize=(6,4))

	hstyle1 = {'stat':'count', 'alpha':0.2, 'element':'step'}
	hstyle2 = {'stat':'count', 'alpha':0.4, 'element':'step'}

	sns.histplot(wvel, ax=ax, **hstyle1)

	sns.histplot(cvel, ax=ax, color="salmon", **hstyle2)

	ax.set_xlabel("trajectory mean velocity $(\mu m/s)$", fontsize=24)

	ax.legend(["selected walking", "selected crawling"])

pub.save_figure("trajectory_mean_velocity_distribution", "paper")




# %%
# ~ PAPER FIGURES
for key in df:
	with mpl.rc_context({"font.size" : 14}):
		fig, ax = plt.subplots(figsize=(4,4))
		sns.histplot(df[key])
		ax.set_xlabel(key)


# %%
valid = ~np.logical_or.reduce([np.isnan(df[key]) for key in df])
print("{} invalid".format((~valid).sum()))


# %%

require_duration = 100
require_nsteps = 100
require_displacement = 3 
# require_walking_displacement = 1
require_aspect = 1.8 

short = np.logical_or.reduce([~valid, df["duration"] < require_duration, df["nsteps"] < require_nsteps])
high_aspect = df["min_aspect"] > require_aspect
high_displacement = df["displacement"] > require_displacement

crawling = np.logical_and.reduce([~short, high_aspect, high_displacement])
walking = np.logical_and.reduce([~short, ~high_aspect, high_displacement])
trapped = np.logical_and.reduce([~short, high_aspect, ~high_displacement])
wtrapped = np.logical_and.reduce([~short, ~high_aspect, ~high_displacement])
horizontal = np.logical_or(trapped, crawling)

print("identified {} short trajectories".format(np.sum(short)))
print("identified {} crawling trajectories".format(np.sum(crawling)))
print("identified {} walking trajectories".format(np.sum(walking)))
print("identified {} horizontal trapped trajectories".format(np.sum(trapped)))
print("identified {} walking trapped trajectories".format(np.sum(wtrapped)))


# %%
print("fraction", 1902/3113)
# * what fraction of the total duration is in the short trajectories
good = df.iloc[~short]["duration"].sum()/df["duration"].sum()
print("duration and nstep threshold selects {:.3f} of the data by duration".format(good))
# df.iloc[short]["duration"].sum()

# %%
# plot the qhat vs displacement for crawling and trapped trajectories
df.iloc[trapped]["qhat"].mean()
df.iloc[crawling]["qhat"].mean()
hdf = df.iloc[horizontal]
# sns.histplot(data=hdf, x="meanvel", y="qhat")
sns.scatterplot(data=df.iloc[crawling], x="meanvel", y="qhat", alpha=0.2)
sns.scatterplot(data=df.iloc[trapped], x="meanvel", y="qhat", alpha=0.2)

# hdf[hdf["first_tangent_corr"] < 0].index

# %%
# extract an interesting trajectory with -ve qhat
df.iloc[trapped].sort_values("qhat")
# interesting track = 487

root = join(pili.root, "notebook")
def animate_idx(idx, targetdir='animate/new'):
	targetdir = join(root, targetdir)
	if not os.path.exists(targetdir):
		os.mkdir(targetdir)
	
	form = join(targetdir, 'track_{:04d}.mp4')
	sample = 10
	for track_idx in idx:
		track = _fj.trackload([track_idx])[0]
		fig = plt.figure()
		target = join(targetdir, form.format(track_idx))
		print("write animation to target")
		twanimation.outline(fig, [track], sample, savefile=target, fps=20)

if animate:
	animate_idx(df.iloc[trapped].index, targetdir='animate/trapped')

# %%

# %%
# plot the linearised velocity of crawling bacteria
fig, ax = plt.subplots(figsize=(4,4))
xlim = (0, 0.20)
sns.histplot(df["lvel"][crawling], binrange=xlim)

# plot persistence
fig, ax = plt.subplots(figsize=(4,4))
sns.histplot(df["qhat"][crawling])
# * expect positive persistencea but persistence can be very low

# %%
# * sort out a low lvel trajectory
crawling_df =  df.iloc[crawling]
sort_lvel = crawling_df.sort_values("lvel")
ref_lvel = sort_lvel.iloc[0]["lvel"]
target_idx = 868
# investigate trajectory 868
# i.e. animate with command
# python -m fj animate_one 868

# %%
# check qhat calculation
check_track_idx = 1519
# qhat
lintrack = _fj.lintrackload([check_track_idx])[0]
original = _fj.trackload_original([check_track_idx])[0]
print("load track ", check_track_idx)

_local = twanalyse._qaparams([lintrack], vquantile=0.95)
print('qhat', _local['qhat']['estimate'])

fig, ax = plt.subplots(figsize=(6,4))
shapeplot.longtracks(ax, [lintrack])

# twanalyse
def _qaparams(ltrs, vquantile=0.99):
	sd = {}
	step_vel = [ltr.get_step_velocity() for ltr in ltrs]
	sim_ut = []
	sim_up = []
	for svel in step_vel:
		sim_ut.append(svel[1:])
		sim_up.append(svel[:-1])
	totalut = np.concatenate(sim_ut)
	totalup = np.concatenate(sim_up)
	# velocity threshold is needed for fanjin data since 
	# there are some huge velocities (probably errors) and these effect the estimate significantly
	th = None
	if vquantile:
		# note: not very efficient
		norm = np.linalg.norm
		s_t = norm(totalut,axis=1)
		s_p = norm(totalup,axis=1)
		th = np.quantile(s_t, vquantile)
		select = np.logical_and(s_t < th, s_p < th)
		totalut = totalut[select]
		totalup = totalup[select]

	return totalut, totalup


u_t, u_p = _qaparams([lintrack]) 
ctp = (u_t*u_p).sum(axis=1)

fig, ax= plt.subplots(figsize=(6,4))
ax.plot(ctp, marker='o', linestyle='--')
print(u_t[-3:])
print(u_p[-3:])

fig, ax= plt.subplots(figsize=(6,4))
ax.plot(lintrack.get_step_speed(), marker='o', linestyle='--')
ax.set_ylim((0,0.1))

fig, ax= plt.subplots(figsize=(6,4))
sns.histplot(lintrack.get_step_speed(), binrange=(0, 0.1))

fig, ax= plt.subplots(figsize=(6,4))
sns.histplot(original.get_speed(), binrange=(0, 1.0))

# the qhat statistics (and therefore ahat as well) are really not robust on a single-trajectory basis
# lets try and replace it with tangent correlation function for linear steps
# plus an analysis of the speed distribution that we are working on 

# %%

_DT = 0.1
# def scaling_tangent(x, max_tau=10):
#     max_shift = int(max_tau/_DT)

#     corr = []
#     for shift in range(1, max_shift):
#         dx = x[shift:] - x[:-shift]
#         ndx = dx/norm(dx,axis=1)[:,np.newaxis]
#         val = np.nanmean( (ndx[shift:]*ndx[:-shift]).sum(axis=1) )
#         corr.append(val)
#     return corr

def scaling_tangent(x, max_tau=10):
	max_shift = int(max_tau/_DT)

	corr = []
	for shift in range(1, max_shift):
		dx = x[shift:] - x[:-shift]
		ndx = dx/norm(dx,axis=1)[:,np.newaxis]
		val = np.nanmean( (ndx[shift:]*ndx[:-shift]).sum(axis=1) )
		corr.append(val)
	return corr


# TODO don't sum overlapping segments

def tangent(tr):
	dx = tr.get_dx()
	return _tangent(dx)

support, line = _tangent(lintrack.get_step_dx())
# line = scaling_tangent(lintrack.get_head2d())

def plot_correlation(support, line):
	ax.plot(support, line, marker='o')
	ax.set_xlabel(r"$\mu m$")
	ax.set_ylim(-1,1)
	ax.axhline(0, c='k', alpha=0.4)

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(6,4))
	plot_correlation(support, line)



# %%
arc_track_idx = 771

_original = _fj.trackload_original([arc_track_idx])[0]
_lintrack = _fj.lintrackload([arc_track_idx])[0]
# line = scaling_tangent(_original.get_head2d())
# line = tangent(_original)

support, line = _tangent(_lintrack.get_step_dx())
# line = scaling_tangent(lintrack.get_head2d())

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(6,4))
	plot_correlation(support, line)

# %%

mtc = hdf["min_tangent_corr"]

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(6,4))
	sns.histplot(mtc, bins=30, ax=ax)

# fig, ax = plt.subplots(figsize=(6,4))
# sns.scatterplot(data = hdf, x="displacement", y="min_tangent_corr", ax=ax)

# %%
anti_correlated = hdf["min_tangent_corr"] < 0
correlated = hdf["min_tangent_corr"] >= 0
hdf.loc[:,"anti_correlated"] = hdf["min_tangent_corr"] < 0
print("hdf['min_tangent_corr'] < 0 : ", (hdf["min_tangent_corr"] < 0).sum())
print("hdf['min_tangent_corr'] >= 0 : ", (hdf["min_tangent_corr"] >= 0).sum())

hdf[anti_correlated]
# * includes some circle looking tracks and some very slow/trapped 

# %%
# * fit with a circle  
import scipy


data = _original.get_head2d()
# Y = 1/R
def circle_fit(data):
	mx, my = np.mean(data, axis=0)
	def lsq(args, data, output_residuals=False):
		mx, my, Y = args
		m = np.array([mx, my])
		v = data - m
		residual = norm(v, axis=1) - 1/Y
		sqresidual = residual**2
		if output_residuals:
			return np.sum(sqresidual)/len(data), residual
		else:
			return np.sum(sqresidual)/len(data)
	x0 = (mx, my, 1)
	result = scipy.optimize.minimize(lsq, x0, args=(data,))
	result.residuals = lsq(x0, data, output_residuals=True)[1]
	return result

result = circle_fit(data)

mx, my, Y = result.x

# plot track using scatter
def plot_data(ax, data, linekw={}):
	x, y = data.T
	ax.plot(x, y, linestyle='none', marker='o', markersize=1, **linekw)
	plotutils.prettyax_track(ax, trs)
	ax.set_aspect('equal')
	return ax

def plot_circle(ax, mx, my, Y):
	R = 1/Y
	theta = np.linspace(0,2*np.pi,360)
	x, y = mx + R*np.cos(theta), my + R*np.sin(theta)
	ax.plot(x, y, linestyle='--', linewidth=4)


fig, ax = plt.subplots(figsize=(5,5))

plot_data(ax, data)
plot_circle(ax, *result.x)

# fig, ax = plt.subplots(figsize=(6,4))
# sns.histplot(result.residuals)

# %%

# %%

def cache_circle_fit(idx, circle_fit_result):
	target_dir = join(pili.root, 'notebook/circle_fit/result')
	if not os.path.exists(target_dir):
		os.makedirs(target_dir)
	form = 'circle_fit_{:04d}.pkl'
	for i, index in enumerate(idx):
		target = join(target_dir, form.format(index))
		with open(target, 'wb') as f:
			pickle.dump(circle_fit_result[i], f)

from glob import glob
def load_circle_fit():
	target_dir = join(pili.root, 'notebook/circle_fit/result/*')
	result_data = []
	for target in sorted(glob(target_dir)):
		with open(target, 'rb') as f:
			result = pickle.load(f)
			# print(target)
			# print(result)
			result_data.append(result)
	return result_data


work = False
if work:
	circle_fit_result = []
	# for index in tqdm(hdf.index):
	for index in tqdm(range(len(original_trs))):
		data = original_trs[index].get_head2d()
		result = circle_fit(data)
		circle_fit_result.append(result)

	cache_circle_fit(range(len(original_trs)), circle_fit_result)
else:
	circle_fit_result = load_circle_fit()



# %%
fit = np.array([result.fun for result in circle_fit_result])
Y = np.array([result.x[2] for result in circle_fit_result])
R = 1/Y
# circular = np.argwhere(R < 3).ravel()
df.loc[:,"circle_fit"] = fit
df.loc[:,"circle_fit_R"] = R
hdf = df.iloc[horizontal]

fig, ax = plt.subplots(figsize=(6,4))
sns.scatterplot(data=hdf, x='circle_fit_R', y='circle_fit', ax=ax)
ax.set_ylim(0,1.0)
ax.set_xlim(0,3.0)

# order, most circular looking first

# %%

def plot_circle_fit(hdf, target_dir='notebook/circle_fit/sort'):

	target_dir = join(pili.root, target_dir)
	if not os.path.exists(target_dir):
		os.makedirs(target_dir)

	form = "lt_{:04d}.png"
	for i, index in tqdm(enumerate(hdf.index)):
		target = join(target_dir, form.format(i))
		data = original_trs[index].get_head2d()
		result = circle_fit_result[index]

		fig, ax = plt.subplots(figsize=(5,5))
		plot_data(ax, data)
		plot_circle(ax, *result.x)
		fig.savefig(target)
		plt.close()

if animate:
	hdf['slice_index'] = range(len(hdf))
	# plot_circle_fit(hdf.sort_values("circle_fit_R"))
	plot_circle_fit(hdf.sort_values("circle_fit"), target_dir='notebook/circle_fit/sort_fit')
	
# %%
# try to pick two statistics that identify the circular data
hdf.sort_values("circle_fit")
fig, ax = plt.subplots(figsize=(6,4))
sns.scatterplot(data=hdf, x="circle_fit",  y=1/hdf["circle_fit_R"], ax=ax, s=2)
ax.set_xlim(0,1.0)
ax.set_ylabel('fit 1/R')
# ax.set_ylim(0,0.1)

# %%

fig, ax = plt.subplots(figsize=(6,4))
sns.scatterplot(data=hdf, x="circle_fit",  y="step_displacement", ax=ax, s=2)
ax.set_xlim(0,1.0)

# %%
# compute the normal of trajectories as curves
P = np.array([[0, -1], [1, 0]])
def tangent_normal(ltr):
	dx = ltr.get_step_dx()
	ndx = dx/norm(dx,axis=1)[:,np.newaxis]
	curvesgn = np.sign(np.cross(ndx[1:], ndx[:-1]))
	n = [P @ d for d in ndx]
	def normal(i):
		sgn = curvesgn[i]
		_normal = (sgn * n[i] + sgn * n[i+1])/2
		return _normal
	return [normal(i) for i in range(len(ndx)-1)]

#  should be the center of the circle
def circle_center_normal(ltr, result):
	center = result.x[:2]
	data = ltr.get_step_head()
	v = data - center
	return v/norm(v,axis=1)[:,np.newaxis]

def center_vector(ltr):
	data = ltr.get_step_head()
	center = np.mean(data, axis=0)
	v = data - center
	return v


def circle_corr(ltr):
	dx = ltr.get_step_dx()
	ndx = dx/norm(dx,axis=1)[:,np.newaxis]
	data = ltr.get_step_head()
	center = np.mean(data, axis=0)
	v = data - center
	norm_v = v/norm(v,axis=1)[:,np.newaxis]
	circ_tangent = [P @ u for u in norm_v]

	if len(dx) > 0: 
		return sum([min(abs((ndx[i] * circ_tangent[i]).sum()), abs((-ndx[i] * circ_tangent[i]).sum()) ) for i in range(len(dx))])/len(dx)
	else:
		return 0.


ltr = ltrs[771]
# ltr = ltrs[1519]
# ltr = ltrs[175]

ex = [771, 1519, 175]

def angle(normal):
	return np.array([np.arctan2(n[1],n[0]) for n in normal])

def travel(normal):
	return np.arccos((normal[1:]*normal[:-1]).sum(axis=1))

def max_travel(ltr, result, normal):
	sgn = np.sign
	# need to know if the path crossed the -ve x plane 
	center = result.x[:2]
	def cross_nx(ltr):
		data = ltr.get_step_head()
		# center the data on the origin
		data = data - center
		N = len(data)
		nx_condition = np.array([data[i][0] < 0 and data[i+1][0] < 0 for i in range(N-1)])
		y_condition = np.array([sgn(data[i][1])*sgn(data[i+1][1]) == -1 for i in range(N-1)])
		# print(nx_condition)
		# print(y_condition)
		return np.any(np.logical_and(nx_condition, y_condition))

	lab_angle = angle(normal)
	
	cross = cross_nx(ltr)
	# print(lab_angle.max(), lab_angle.min(), cross)
	# print(lab_angle)
	# print(lab_angle[lab_angle < 0].max())
	# print(lab_angle[lab_angle > 0].min()
	if cross:
		a = lab_angle[lab_angle < 0].max()
		b = lab_angle[lab_angle > 0].min()
		return (np.pi - b) + (np.pi - (-a))
	else:
		return lab_angle.max() - lab_angle.min()

# ltr = ltrs[ex[2]]
# track_idx = 2140
track_idx = ex[0]
ltr = ltrs[track_idx]

result = circle_fit_result[track_idx]
normal = circle_center_normal(ltr, result)
mtr = max_travel(ltr, result, normal)
"{:.1f}".format(180/np.pi * mtr)

# %%
hdf_circle_normal = [circle_center_normal(ltrs[i], circle_fit_result[i]) for i in hdf.index]
hdf_total_travel = [np.sum(np.abs(travel(normal))) for normal in hdf_circle_normal]

# %%

hdf_max_travel = np.empty(len(hdf))
for i, hdf_i in enumerate(hdf.index):
	# print('track', hdf_i)
	value = max_travel(ltrs[hdf_i], circle_fit_result[hdf_i], hdf_circle_normal[i]) 
	hdf_max_travel[i] = value

# %%
hdf.loc[:,"travel"]  = hdf_total_travel/hdf_max_travel
hdf.sort_values("travel", ascending=False)

# %%

for track_idx in ex:
	print('track ', track_idx)
	ltr = ltrs[track_idx]
	result = circle_fit_result[track_idx]
	c_normal = circle_center_normal(ltr, result)
	print('center angle variance ', np.var(angle(c_normal)))
	print('center angle travel/micron ', travel(c_normal).sum()/ltr.get_total_length())
	print('circle correlation', circle_corr(ltr))

# travel_rate = np.array([travel(circle_center_normal(ltr)).sum()/ltr.get_total_length() for ltr in ltrs])
# print(travel_rate)


# %%

c_circle = np.array([circle_corr(ltr) for ltr in ltrs])

# if animate:
if True:
	hdf['slice_index'] = range(len(hdf))
	hdf["c_circle"] = c_circle[hdf.index.to_numpy()]
	# plot_circle_fit(hdf.sort_values("travel_rate"), target_dir='notebook/circle_fit/travel_sort')
	# plot_circle_fit(hdf.sort_values("c_circle"), target_dir='notebook/circle_fit/circle_sort')
	# plot_circle_fit(hdf.sort_values("c_circle"), target_dir='notebook/circle_fit/circle_sort')

	plot_circle_fit(hdf.sort_values("travel", ascending=False), target_dir='notebook/circle_fit/travel_sort')

# %%
# Annotate circular using sort_values("travel", ascending=False)
# using 'travel' statistic
annotate_circle_anomaly = list(range(37+1)) + [39, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 76, 81]
# convert to trackidx
annotation = hdf.sort_values("travel", ascending=False).index[annotate_circle_anomaly]
column = np.full(len(df), False)
column[annotation] = True
df.loc[:, "circle_anomaly"] = column
hdf.loc[:, "circle_anomaly"] = column[hdf.index]

# %%
df.to_pickle("classification.pkl")
# save again

# %%


target = join(pili.root, "notebook/thesis_classification/kmsd.npy")
if work:
	kmsd = np.array([twanalyse.kmsd(tr) for tr in tqdm(original_trs)])
	np.savetxt(target, kmsd)
else:
	kmsd = np.loadtxt(target)

# %%

np.mean(kmsd[crawling])
np.mean(kmsd[walking])
# np.mean(kmsd[hdf.index])

# %%
# expect circular trajectories to have low kmsd
# also we can separate out trapped trajectories as been on the low end of the kmsd histogram

df.loc[:,"kmsd"] = kmsd
hdf.loc[:, "kmsd"] = kmsd[hdf.index]

fig, ax = plt.subplots(figsize=(5,5))
sns.histplot(hdf["kmsd"], ax=ax)
ax.set_xlabel("kmsd")

if animate:
	plot_circle_fit(hdf.sort_values("kmsd"), target_dir='notebook/circle_fit/sort_kmsd')


# %%

# xdata = hdf["kmsd"]
xdata = hdf["displacement"]/hdf["step_displacement"]
index = hdf["circle_anomaly"]

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(5,4))
	# ax.set_ylim((0,60))
	ax.set_yscale('log')
	mstyle = {"alpha":0.5}

	ax.scatter(xdata[~index], hdf["travel"][~index], **mstyle)
	ax.scatter(xdata[index], hdf["travel"][index], **mstyle)
	
	ax.set_ylabel(r"$\theta_T / \theta_{\mathrm{arc}}$", fontsize=28)

	# ax.set_xlabel(r"$k_{\mathrm{MSD}}$")
	ax.set_xlabel(r"$h/C$", fontsize=28)

	ax.legend(["typical", "circular anomaly"])
	# ax.xaxis.set_ticks([0,0.5,1.0,1.5, 2.0])
	
	plt.draw()
	ax.set_xticklabels([t.get_text().replace('mathdefault', 'mathbf') for t in ax.xaxis.get_ticklabels()])
	ax.set_yticklabels([t.get_text().replace('mathdefault', 'mathbf') for t in ax.yaxis.get_ticklabels()])

# thesis.save_figure("circle_anomaly_scatter")

# %%


# %%
# todo : plot mean squared displacement curves for circular and diffusive trajectories

def compute_msd(ltr, scaling):
	x = ltr.get_step_head()
	msd_n = [i for i in range(len(scaling))]
	for i, shift in enumerate(scaling):
		dx = norm(x[shift:] - x[:-shift], axis=1)
		msd_n[i] = np.mean(dx**2)
	return scaling, msd_n

fig, ax = plt.subplots(figsize=(6,4))
ex_label = ["circular", "not circular", "not circular"]
for i ,track_idx in enumerate(ex):
	ltr = ltrs[track_idx]

	scaling = list(range(1,100))
	scaling, msd = compute_msd(ltr, scaling)
	ax.loglog(scaling, msd, label=ex_label[i])

ax.legend()

# %%
# TODO full table

df.keys()

# %%
np.sum(walking)
np.sum(wtrapped)
walkingidx = _fj.load_subset_idx()["walking"]
# convert to 3113 array
select_walking = np.zeros(len(df), dtype=bool)
select_walking[walkingidx] = 1
filter_wa = select_walking & walking


# %%

# * count overlap between circle anomaly and crawling 

anomaly = df["circle_anomaly"]
np.sum( df["circle_anomaly"] & crawling ), np.sum( df["circle_anomaly"] & trapped )
cr_anom = df["circle_anomaly"] & crawling
tr_anom = df["circle_anomaly"] & trapped
filter_cr = crawling ^ cr_anom
filter_tr = trapped ^ tr_anom
print('kmsd', np.mean(kmsd[filter_wa]), np.mean(kmsd[filter_cr]))
print(np.sum(crawling), np.sum(filter_cr))
print(np.sum(trapped), np.sum(filter_tr))


subset = [filter_cr, filter_wa, filter_tr, wtrapped, anomaly, short]
namelist = ["crawling", "walking", "trapped", "trapped-walking", "anomaly", "short"]
keys = ["class", "count", "median_step", "median_velocity"]
headers = ["class", "count", "median_step", "median_velocity"]

_data = {key: [] for key in keys}

for name, index in zip(namelist, subset):
	sub = df[index]
	duration = sub["duration"]
	lvel = sub["lvel"]

	sub_tr = [ltrs[i] for i, c in enumerate(index) if c]
	stepdt = np.concatenate([tr.get_step_dt() for tr in sub_tr])

	_data["class"].append(name)
	_data["count"].append(len(sub))
	_data["median_step"].append( np.median(stepdt) )
	_data["median_velocity"].append( lvel.median() )

classdf = pd.DataFrame(_data)
classdf

# TODO convert this table into latex

# %%

classdf['count'].sum()



# %%
# * SAVE
# save the subsets to be used elsewhere

for name, sub in zip(namelist, subset):
	f = join(pili.root, 'notebook/thesis_classification', name+'.npy')
	track_index = np.argwhere(np.array(sub)).ravel().astype(dtype=int)
	print('writing to ', f)
	np.savetxt(f, track_index)



# target = join(pili.root, "notebook/thesis_classification/kmsd.npy")


	
# %%
# * we can draw tables but perhaps drawing distributions is bettter?
datalist = [None for _ in range(len(ltrs))]
for i, ltr in enumerate(ltrs):
	if ~short[i]:
		datalist[i] = twanalyse.observables([ltr])

# %%

shortlist = ["crawling", "walking", "trapped"]
shortset = [subset[i] for i, name in enumerate(namelist) if name in shortlist]

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
color = iter(defcolor)
hstyle = {'stat':'count', 'alpha':0.3, 'element':'step'}

with mpl.rc_context({"font.size" : 20, "text.usetex": True}):
	fig, ax = plt.subplots(figsize=(6,4))

	leg = []

	for name, index in zip(namelist, shortset):
		if name == 'short': continue
		sub_data = [datalist[i] for i, c in enumerate(index) if c]
		sub_tr = [ltrs[i] for i, c in enumerate(index) if c]
		
		vel = [ld["lvel"]["mean"] for ld in sub_data]
		print(name)
		print('median', np.median(vel))
		xlim = (0,0.5)
		sns.histplot(vel, ax=ax, color=next(color), binrange=xlim, **hstyle)

		leg.append(name)

	ax.set_xlabel(r"trajectory mean velocity (\textmu m)", fontsize=24)
	ax.legend(leg)

thesis.save_figure("mean_lvel_by_class")

# %%
# plot kmsd

hstyle = {'stat':'count', 'alpha':0.1, 'element':'step', 'linewidth':2}
color = iter(defcolor)

with mpl.rc_context({"font.size" : 20, "text.usetex": True}):
	fig, ax = plt.subplots(figsize=(6,4))

	leg = []

	for name, index in zip(namelist, shortset):
		if name == 'short': continue
		sub_data = [kmsd[i] for i, c in enumerate(index) if c]
		print(name, 'kmsd', np.median(sub_data))
		
		# xlim = (0,0.5)
		xlim = (0,2.0)
		bins = np.linspace(0,2.0,20)
		sns.histplot(sub_data, ax=ax, color=next(color), binrange=xlim, bins=bins, **hstyle)

		leg.append(name)

	ax.set_xlabel("$k_{\mathrm{MSD}}$", fontsize=24)
	ax.legend(leg, loc='upper left')

thesis.publication.save_figure("kmsd_by_class")


# %%

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
color = iter(defcolor)
hstyle = {'stat':'count', 'alpha':0.3, 'element':'step'}

with mpl.rc_context({"font.size" : 20, "text.usetex": True}):
	fig, ax = plt.subplots(figsize=(6,4))

	leg = []

	for name, index in zip(namelist, shortset):
		if name == 'short': continue
		_hstyle = hstyle.copy()
		if name == 'walking':
			_hstyle["bins"] = 5
		sub_data = [datalist[i] for i, c in enumerate(index) if c]
		sub_tr = [ltrs[i] for i, c in enumerate(index) if c]
		
		stepdt = [np.median(tr.get_step_dt()) for tr in sub_tr]
		sns.histplot(stepdt, ax=ax, color=next(color), **_hstyle)

		leg.append(name)

	ax.set_xlabel("trajectory median step time (s)", fontsize=24)
	ax.legend(leg)

thesis.publication.save_figure("mean_stepdt_by_class")


# %%
# same for variance of deviation angle, kmsd, and q/a statistics?



# %%
