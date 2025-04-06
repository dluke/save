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
# construct figures for a presentation on the EM-paper
# its only sensible to keep this separate from em_algorithm.py

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
pub.set_write_dir("/home/dan/usb_twitching/sparseml/EM_paper/impress")
print("writing figures to", pub.writedir)

# import pili.publication as thesis
# thesis.set_write_dir("/home/dan/usb_twitching/thesis")
# print(thesis.writedir)

import thesis.publication as thesis


import pili
from pili import support
import emanalyse
import _fj
import mdl
import pwlpartition

import fjanalysis
import twanalyse
import pwlstats

from skimage.restoration import denoise_wavelet
from skimage.restoration import  estimate_sigma

import sklearn.mixture

# %% 
mplstyle = {"font.size": 20}
texstyle = {"font.size": 20, "text.usetex": True, "axes.labelsize" : 24}
vstyle = dict(alpha=0.3,c='black',linestyle='--', lw=3)
notename = "em_algorithm"
publish = False

# %%
# load dflist
from glob import glob
q2 = 'notebook/em_algorithm/crawling_em_quad_2/'
q3 = 'notebook/em_algorithm/crawling_em_quad_3/'

qpath = q2

dflist = [pd.read_pickle(at) for at in sorted(glob(join(pili.root, qpath, 'df_*.pkl')))]

# load the datalist
auxdir = join(pili.root, 'notebook/em_algorithm/crawling_em_quad_2/', 'aux')

with open(join(pili.root, auxdir, 'crawling_udatalist.pkl'), 'rb') as f:
	udatalist = pickle.load(f)

with open(join(pili.root, qpath, 'aux', 'crawling_datalist.pkl'), 'rb') as f:
	datalist = pickle.load(f)


# %%
# load the metadata aswell
load_path = join(pili.root, 'notebook/thesis_classification/crawling.npy')
idx_list = np.loadtxt(load_path).astype(int)
track_list = _fj.trackload_original(idx_list)
ltr_list = _fj.lintrackload(idx_list)
wave_list = _fj.trackload(idx_list)

ldata = fjanalysis.load_summary()
topdata = [ldata[i] for i in idx_list]
vel = np.array([ld['lvel']['mean'] for ld in topdata])
print(vel.size)


vel_order = np.argsort(vel)
vel_order[-1]

vel.min(), vel.max()

# %%
# plot the step time distribution
# sliding window
# iwindow = [-10, -1]
# iwindow = [-100,-80]
# iwindow = [0, 20]
iwindow = [250,300]
ix, iy = iwindow
index = vel_order[-1]
# dt = np.concatenate([ltr_list[vel_order[i]].get_step_dt() for i in range(ix, iy+1)])
dt = np.concatenate([_fj.linearize(track_list[vel_order[i]], step_d=0.06).get_step_dt() for i in range(ix, iy+1)])
_xlim = [0,50]
_xlim = [0,20]
sns.histplot(dt, binrange=_xlim)

# %%

# setup the code for computing mapped velocities
wavelet='db1'
em_config = {"wavelet":wavelet, 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}
# wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
def get_data(wavemodel, lptr):
	curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr)
	udata = np.diff(curve_coord) 
	return udata

track = track_list[654]
wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
def coarsen(lptr, n):
	# if lptr.dt == None:
	# 	lptr.reset_dt()
	N = lptr.M
	lptr = lptr.copy()
	lptr.x = np.array([lptr.x[i:i+n].mean() for i in range(0, N-n)])
	lptr.y = np.array([lptr.y[i:i+n].mean() for i in range(0, N-n)])
	lptr.M = len(lptr.x)
	return lptr



# %%

index = -12
index = 694
index = 654
index = vel_order[-50]
v = vel[index]
# sns.histplot(vel)
track = track_list[index]
absv = track.get_head_v()
abs_speed = np.linalg.norm(absv, axis=1)
# sns.histplot(abs_speed)
# coarse grain with sliding window
x, y = track['x'], track['y']

wsize = 1


with mpl.rc_context(mplstyle):

	fig, ax = plt.subplots(figsize=(4,4))

	# for wsize in [1,3,5,10]:
	for wsize in [1]:
		xw = np.array([x[i:i+wsize].mean() for i in range(0, x.size-wsize)])
		yw = np.array([y[i:i+wsize].mean() for i in range(0, y.size-wsize)])

		vw = np.sqrt( (xw[1:] - xw[:-1])**2 +  (yw[1:] - yw[:-1])**2 ) 

		style = dict(element="step", fill=False, alpha=0.5)
		sns.histplot(vw, binrange=(0,0.08), ax=ax, lw=5, **style)
	ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))
	ax.set_xlabel(r"displacement $\mu m$")
pub.save_figure('unsigned_disp')

# %%
# * MAP -> coarse grain

from matplotlib.ticker import ScalarFormatter

data = datalist[index]
udata = udatalist[index]

# use processed or unprocessed data?
# _data = data
_data = udata

print('count zeros', (_data==0).sum()/_data.size)

wsize = 3
dw = np.array([_data[i:i+wsize].mean() for i in range(0, _data.size-wsize)])

shstyle = dict(element="step", fill=False, alpha=0.6)
with mpl.rc_context(mplstyle):
	_xlim = (-0.05, 0.05)
	fsize = (8,8)
	fsize = (4,4)

	fig, ax = plt.subplots(figsize=fsize)
	_vstyle = vstyle.copy()
	_vstyle["lw"] = 2
	ax.axvline(0, **_vstyle)
	ax.axvline(0.03, **_vstyle)
	ax.axvline(-0.03, **_vstyle)

	# ax.axvline(0.03/wsize, **vstyle)

	# sns.histplot(data, ax = ax, lw=4, **shstyle)

	_data = _data[_data!=0]
	sns.histplot(_data, ax = ax, lw=5, **shstyle)

	# sns.histplot(dw, ax = ax, lw=4, **shstyle)
	ax.set_xlim(_xlim)

	# ax.xaxis.set_major_formatter( ScalarFormatter(useOffset=False) )
	# ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	ax.xaxis.set_ticks([-0.03, 0, 0.03])
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))

	ax.set_xlabel(r"displacement $(\mu m)$")



pub.save_figure('signed_disp')

# %%

t = 0.03
delta = 0.01

window = t-delta, t+delta
plus = np.logical_and(data > window[0], data < window[1])
window = -t-delta, -t+delta
minus = np.logical_and(data > window[0], data < window[1])
print(plus.sum(), minus.sum(), data.size)

char = np.full_like(data, '.', dtype=object)
char[plus] = '+'
char[minus] = '-'
''.join(char.tolist())[:100]

# %%
# * coarse grain -> MAP
wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
coarse_lptr = coarsen(lptr, wsize)
coarse_curve_coord = emanalyse.compute_curve_coordinate(wavemodel, coarse_lptr)
coarse_udata = np.diff(coarse_curve_coord) 


shstyle = dict(element="step", fill=False, alpha=0.6)
with mpl.rc_context(mplstyle):
	_xlim = (-0.05, 0.05)

	fig, ax = plt.subplots(figsize=fsize)
	ax.axvline(0, **vstyle)
	ax.axvline(-0.03, **vstyle)
	ax.axvline(0.03, **vstyle)
	sns.histplot(coarse_udata, ax = ax, lw=4, **shstyle)
	h1 = ax.lines[-1]
	sns.histplot(dw, ax = ax, lw=4, **shstyle)
	h2 = ax.lines[-1]
	ax.set_xlim(_xlim)
	# ax.ticklabel_format(axis='x', useOffset=1e-2)
	ax.legend([h1, h2], ['C', 'M'])




# %%
from matplotlib.animation import FuncAnimation

with mpl.rc_context(mplstyle):

	fig, ax = plt.subplots(figsize=(5,4))

	w_list = [1,3,5,10]

	style = dict(element="step", fill=False, alpha=0.5)

	handles = []
	labels = []

	ax.set_ylim(0,550)
	xlim = (0, 0.10)
	ax.set_xlim(xlim)
	ax.set_xlabel(r"mean displacement $(\mu m)$")

	ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))

	def update(frame):
		print('frame', frame)
		wsize = w_list[frame]
		
		xw = np.array([x[i:i+wsize].mean() for i in range(0, x.size-wsize)])
		yw = np.array([y[i:i+wsize].mean() for i in range(0, y.size-wsize)])
		vw = np.sqrt( (xw[1:] - xw[:-1])**2 +  (yw[1:] - yw[:-1])**2 ) 

		sns.histplot(vw, binrange=xlim, ax=ax, lw=3, **style)
		handles.append(ax.lines[-1])
		labels.append(r"$\Delta t = {:.1f}$".format(0.1* wsize))
		ax.legend(handles, labels, fontsize=18, loc='upper right', bbox_to_anchor=(1.04,1))
		plt.tight_layout()

	def init():
		pass

	ani = FuncAnimation(fig, update, frames=list(range(0, len(w_list))), 
		interval=1000, init_func=init)
	

	present_dir = join(pili.root, '../sparseml/EM_paper/impress/ims')
	savefile = join(present_dir, 'xdisp_01.avi')
	print('save at', savefile)
	ani.save(savefile)

# %%

with mpl.rc_context(mplstyle):

	fig, ax = plt.subplots(figsize=(5,4))

	w_list = [1,3,5,10]

	style = dict(element="step", fill=False, alpha=0.5)

	handles = []
	labels = []

	ax.set_ylim(0,550)
	ax.set_xlim(xlim)
	ax.set_xlabel(r"total displacement $(\mu m)$")

	ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))


	def update(frame):
		print('frame', frame)
		wsize = w_list[frame]
		
		xw = np.array([x[i:i+wsize].sum() for i in range(0, x.size-wsize)])
		yw = np.array([y[i:i+wsize].sum() for i in range(0, y.size-wsize)])
		vw = np.sqrt( (xw[1:] - xw[:-1])**2 +  (yw[1:] - yw[:-1])**2 ) 

		sns.histplot(vw, binrange=xlim, ax=ax, lw=3, **style)
		handles.append(ax.lines[-1])
		labels.append(r"$\Delta t = {:.1f}$".format(0.1* wsize))
		ax.legend(handles, labels, fontsize=18)
		plt.tight_layout()

	def init():
		pass

	ani = FuncAnimation(fig, update, frames=list(range(0, len(w_list))), 
		interval=1000, init_func=init)
	

	present_dir = join(pili.root, '../sparseml/EM_paper/impress/ims')
	savefile = join(present_dir, 'xtdisp_01.avi')
	print('save at', savefile)
	ani.save(savefile)

# %%
# pub context plot

class Mixture(object):

	def __init__(self, mix, weights):
		self.mix = mix
		self.weights = weights

	def pdf(self, x):
		res = np.zeros_like(x)
		for w, mix in zip(self.weights, self.mix):
			res += w * mix.pdf(x)
		return res


def reconstruct_nntn(em):
	m0, sigma, m1, sigma1, lam = em['parameters']
	w1, w2 = em['weights']
	mix = [emanalyse.Gauss(m0, sigma), emanalyse.NTN(m1, sigma, sigma1)]
	construct = Mixture(mix, em['weights'])
	return construct

def reconstruct_gemg(em):
	m0, sigma, m1, sigma, lam = em['parameters']
	w1, w2 = em['weights']
	mix = [emanalyse.Gauss(m0, sigma), emanalyse.EMG(m1, sigma, lam)]
	construct = Mixture(mix, em['weights'])
	return construct
	
def reconstruct_leml(em):
	m0, scale, m1, lam1, lam2 = em['parameters']
	w1, w2 = em['weights']
	mix = [emanalyse.Laplace(m0, scale), emanalyse.EML(m1, lam1, lam2)]
	construct = Mixture(mix, em['weights'])
	return construct

def reconstruct_lltn(em):
	m0, scale, m1, sigma, lam = em["parameters"]
	mix = [emanalyse.Laplace(m0, scale), emanalyse.LTN(m1, sigma, lam)]
	construct = Mixture(mix, em['weights'])
	return construct


def remove_zeros(data):
	return data[data!=0]

def describe_construct(construct):
	print('weights', construct.weights)
	for mix in construct.mix:
		print(mix.describe())

# %%
gemg_index, leml_index, nntn_index, lltn_index = 0, 1, 2, 3

index = -12
index = vel_order[-40]
index = vel_order[-50]

data = datalist[index]

em = dflist[index].iloc[nntn_index]
construct = reconstruct_nntn(em)

# with mpl.rc_context({"font.size": 18}):
with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize = (6,4))
	handles = emanalyse.pub_context_plot(ax, remove_zeros(data), construct)
	ax.legend(handles, ["data", "Normal", r"$f_{\sigma_0} \ast f_{\sigma_1}^+$", "Mixture"], fontsize=20)
	ax.set_xlabel(r"displacement (\textmu m)")

pub.save_figure('g_sigma_sigma')
thesis.save_figure('g_sigma_sigma')

# %%

em = dflist[index].iloc[gemg_index]
construct = reconstruct_gemg(em)

describe_construct(construct)



# with mpl.rc_context({"font.size": 18}):
with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize = (6,4))
	handles = emanalyse.pub_context_plot(ax, remove_zeros(data), construct)
	ax.legend(handles, ["data", "Normal", r"$f_{\sigma} \ast f_{\lambda}$", "Mixture"], fontsize=20)
	ax.set_xlabel(r"displacement (\textmu m)")

pub.save_figure('g_sigma_lambda')

ax.set_ylabel('')
ax.yaxis.set_ticks([])

thesis.save_figure('g_sigma_lambda')

# %%

em = dflist[index].iloc[leml_index]
construct = reconstruct_leml(em)

# with mpl.rc_context({"font.size": 18}):
with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize = (6,4))
	handles = emanalyse.pub_context_plot(ax, remove_zeros(data), construct)
	ax.legend(handles, ["data", "Laplace", r"$f_{|\lambda_0|} \ast f_{\lambda_1}$", "Mixture"], fontsize=20)
	ax.set_xlabel(r"displacement (\textmu m)")

pub.save_figure('g_lambda_lambda')

ax.set_ylabel('')
ax.yaxis.set_ticks([])

thesis.save_figure('g_lambda_lambda')
# %%
np.median(vel)
np.quantile(vel, 0.025), np.quantile(vel, 0.975)

# %%

em = dflist[index].iloc[lltn_index]
construct = reconstruct_lltn(em)

# with mpl.rc_context({"font.size": 18}):
with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize = (6,4))
	handles = emanalyse.pub_context_plot(ax, remove_zeros(data), construct)
	ax.legend(handles, ["data", "Laplace", r"$f_{|\lambda|} \ast f_{\sigma}^+$", "Mixture"], fontsize=20)
	ax.set_xlabel(r"displacement (\textmu m)")

pub.save_figure('g_lambda_sigma')

thesis.save_figure('g_lambda_sigma')


# %%
# * continue by analysing slingshots

# ! find my code that mimics Jin et al. slingshot plots

index = vel_order[-50]
index = vel_order[-100]
track = track_list[index]
ltr = _fj.linearize(track)
print('N', ltr.get_nsteps())

# example 1
twanalyse.plot_actiondata( twanalyse.actions([ltr]) )
# TODO no 0.1 second fast actions, we should try aggressively defining fast actions from large 0.1s displacements
# TODO and the remaineder will be slow actions


# %%

# example 2
fixed_threshold = 0.1

proc_threshold = np.quantile(ltr.get_step_speed(), 0.9)
print('threshold', proc_threshold)

# polardata = twanalyse.allpolar([ltr], vthreshold=fixed_threshold)
polardata = twanalyse.allpolar([ltr], vthreshold=proc_threshold)

fig, axes = plt.subplots(3, 2, subplot_kw=dict(polar=True), figsize=(8,12))
twanalyse.plotpolar(axes, polardata)

# %%
# * POPULATION, i.e. per trajectory
vel_list = [ltr.get_step_speed() for ltr in ltr_list]
disp_list = [ltr.get_step_displacement() for ltr in ltr_list]
fast_idx_list = [speed > fixed_threshold for speed in vel_list]
count_fast = np.array([fv.sum() for fv in fast_idx_list])
count_step = np.array([ltr.get_nsteps() for ltr in ltr_list])

disp_fast = np.array([disp[fv].sum() for disp, fv in zip(disp_list, fast_idx_list)])
disp_total = np.array([disp.sum() for disp in disp_list])

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	sns.histplot(disp_fast/disp_total, ax=ax, binwidth=0.05, binrange=(0,1.0))
	ax.set_xlabel(r"$C[V>0.1] / C$")
	# ax.set_title(r"$V > {:.1f} $ \textmu m/s".format(fixed_threshold))

print('median fraction', np.median(disp_fast/disp_total))
print('mean fraction', np.mean(disp_fast/disp_total))

pub.save_figure("fraction_of_displacement")
thesis.save_figure("fraction_of_displacement")

# %%
# !redo polar histogram plots for deviation angle
def compute_deviation(ltr, vthreshold):

	def angle(dx, dy):
		norm_dx = dx/np.linalg.norm(dx, axis=1)[:,np.newaxis]
		norm_dy = dy/np.linalg.norm(dy, axis=1)[:,np.newaxis]
		theta = np.arccos(np.sum(norm_dx * norm_dy,axis=1))
		theta *= np.sign(np.cross(norm_dx, norm_dy))
		return theta

	dx = ltr.get_step_dx()
	velocity = ltr.get_step_velocity()
	speed = np.linalg.norm(velocity, axis=1)
	fastidx = speed > vthreshold
	dx_theta = np.arctan2(dx[:,1],dx[:,0])
	lead = np.stack([ltr['x'],ltr['y']],axis=1)
	trail = np.stack([ltr['trail_x'],ltr['trail_y']],axis=1)
	b_ax = lead[ltr.step_idx] - trail[ltr.step_idx]
	theta = angle(dx, b_ax[1:])
	return theta
	# return theta[fastidx]

def get_fast_idx(ltr_list, vthreshold):
	v = np.concatenate([ltr.get_step_speed() for ltr in ltr_list])
	return v > vthreshold


# compute crawling
with mpl.rc_context(thesis.texstyle):
	for vt in [0, 0.05, 0.1, 0.3]:
		fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
		shstyle = dict(element="step", fill=False, alpha=0.6, lw=4, stat='density')
		# with mpl.rc_context(thesis.texstyle):
		deviation = np.concatenate([compute_deviation(ltr, vt) for ltr in ltr_list])
		fast_idx = get_fast_idx(ltr_list, vt)
		fast_deviation = deviation[fast_idx]
		f = fast_idx.sum()/fast_idx.size
		print('fraction', f)
		# sns.histplot(deviation, bins=50, ax=ax, **shstyle)
		sns.histplot(fast_deviation, bins=50, ax=ax, **shstyle)
		ax.set_yticks([])
		ax.set_ylabel('')

		ax.set_title(r'$V > {:.2f}$ \textmu m/s'.format(vt))

		if vt == 0.3:
			ax.legend([r"$\theta_d$ distribution"], loc=(1.04,0.5), fontsize=24)
		thesis.save_figure("deviation_angle_distrib_v={:.2f}".format(vt))

# %%

# fast_list = [ltr.get_step_speed() for ltr in ltr_list]
# threshold  = 0.1
# fraction_list = [(v > threshold).sum()/v.size for v in fast_list]

# with mpl.rc_context(thesis.texstyle):
# 	fig, ax = plt.subplots(figsize=(4,4))
# 	sns.histplot(fraction_list, ax=ax, *)
# 	ax.set_xlabel(r"$\Delatx")


# fast_fractions


# sns.histplot

# %%
# ! what about total rotation

def get_step_body_angle(track, step_idx):
	# !coarse grain the body axis over the steps
	step_idx = np.array(step_idx)
	head = np.stack([track["x"], track["y"]], axis=-1)
	tail = np.stack([track["trail_x"], track["trail_y"]], axis=-1)
	body = head - tail
	body_angle = np.arctan2(body[:,1], body[:,0])
	step_body_angle = np.array([body_angle[step_idx[i]:step_idx[i+1]].mean() for i in range(step_idx.size-1)])
	return step_body_angle

step_angle_list = [get_step_body_angle(track_list[i], ltr_list[i].step_idx) for i in range(len(track_list))]
body_delta_list = [np.diff(body_angle) for body_angle in step_angle_list]
# body_delta_list = [np.diff(body_angle) for body_angle in body_angle_list]
total_rotation = np.array([np.abs(body_delta).sum() for body_delta in body_delta_list])
fast_rotation = np.array([np.abs(body_delta[fv[1:]]).sum() for body_delta, fv in zip(body_delta_list, fast_idx_list)])

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	sns.histplot(fast_rotation/total_rotation, ax=ax, binwidth=0.05, binrange=(0,1))
	ax.set_xlabel(r"$\Delta \theta / \theta$")

	ax.set_title("$V > {:.1f} \mu m/s$".format(fixed_threshold))

print('median', np.median(fast_rotation/total_rotation))
print('mean', np.mean(fast_rotation/total_rotation))

pub.save_figure("fast_fraction_of_rotation")


# %%
# ! Thesis figures
pub.set_write_dir(join(pili.root, "../thesis"))

wave = wave_list[index]

def plot_velocity_distrib(data, xlim=(0,1.0), scale='linear'):
	log_scale = True if scale == 'log' else False
	with mpl.rc_context(mplstyle):
		fig, ax = plt.subplots(figsize=(4,4))
		# entries, bins = np.histogram(data, bins='auto')
		sns.histplot(data, ax=ax, log_scale=log_scale, **shstyle)
		ax.set_xlim(xlim)
	ax.axvline(0.3, **vstyle)
		
def overlay_velocity_distrib(ax, data_list, xlim=(0,1.0), scale='linear'):
	shstyle = dict(element="step", fill=False, alpha=0.6, lw=4)
	data1, data2 = data_list
	log_scale = True if scale == 'log' else False

	sns.histplot(data1, ax=ax, log_scale=log_scale, **shstyle)
	sns.histplot(data2, ax=ax, log_scale=log_scale, **shstyle)
	ax.set_xlim(xlim)
	# ax.axvline(0.3, **vstyle)


_scale = 'log'
_xlim = (1e-3, 1)
# plot_velocity_distrib( track.get_speed(), xlim=_xlim, scale=_scale)
# plot_velocity_distrib( wave.get_speed(), xlim=_xlim, scale=_scale)

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	overlay_velocity_distrib(ax, [track.get_speed(), wave.get_speed()], xlim=_xlim, scale=_scale)
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))
	ax.legend(['original', 'denoised'], loc=(0.05,0.86), framealpha=1.0, fontsize=24)
	ax.set_xlabel(r"velocity (\textmu m/s)")
	# ax.axvline(0.3, **vstyle)

# index = 488 (vel_order -50)
pub.save_figure('ch3_example_log_velocity')
thesis.save_figure('ch3_example_log_velocity')

# %%
# concatentate the data

track_distrib = np.concatenate([track.get_speed() for track in track_list])
wave_distrib = np.concatenate([wave.get_speed() for wave in wave_list])
def make_finite(arr):
	return arr[np.isfinite(arr)]
def clip(arr, lim):
	a, b = lim	
	return arr[np.logical_and(arr>a, arr<b)]

track_distrib = clip(make_finite(track_distrib), _xlim)
wave_distrib = clip(make_finite(wave_distrib), _xlim)


# %%

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	overlay_velocity_distrib(ax, [track_distrib, wave_distrib], xlim=_xlim, scale=_scale)
	# ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))
	ax.legend(['original', 'denoised'], loc=(0.05,0.86), framealpha=1.0, fontsize=24)
	ax.set_xlabel(r"velocity (\textmu m/s)")
	ax.axvline(0.3, **vstyle)

pub.save_figure('ch3_all_log_velocity')
thesis.save_figure('ch3_all_log_velocity')


# %%
# concat the udata
all_udata = np.concatenate(udatalist)

# %%

dxlim = (-0.1, 0.1)
dxlim = (-0.08, 0.08)

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	shstyle = dict(element="step", fill=False, alpha=0.6, lw=4)
	sns.histplot(remove_zeros(all_udata), ax=ax, binrange=dxlim, **shstyle)
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))
	ax.axvline(0.0, **vstyle)
	ax.set_xlabel(r"displacement (\textmu m)")

	sf = ScalarFormatter(useOffset=False)
	sf.set_scientific(True)
	sf.set_powerlimits((-1,1))
	ax.yaxis.set_major_formatter( sf )

pub.save_figure('ch3_all_signed_velocity')
thesis.save_figure('ch3_all_signed_velocity')

# %%

with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	shstyle = dict(element="step", fill=False, alpha=0.6, lw=4)
	sns.histplot(remove_zeros(udatalist[index]), ax=ax, binrange=dxlim, binwidth=0.005, **shstyle)
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))
	ax.axvline(0.0, **vstyle)
	ax.set_xlabel(r"displacement (\textmu m)")

pub.save_figure('ch3_example_signed_velocity')
thesis.save_figure('ch3_example_signed_velocity')




# %%
