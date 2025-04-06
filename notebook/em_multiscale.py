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
# the same EM analysis but this time vary the sampling rate of the data 
# the point here is that 
#  1. we can see how parameters vary ~ and hence what sort of timescales exist in the data
#  2. we can show either or not high time resolution experiments are important

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

import pili.publication as thesis
thesis.set_write_dir("/home/dan/usb_twitching/thesis")
print(thesis.writedir)

import pili
from pili import support
import readtrack
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
vstyle = dict(alpha=0.3,c='black',linestyle='--', lw=2)
notename = "em_multiscale"
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

# setup the code for computing mapped velocities
wavelet='db1'
em_config = {"wavelet":wavelet, 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}

def get_data(track):
	wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
	curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr)
	udata = np.diff(curve_coord) 
	return udata, wavemodel, lptr, meta


# %%
idx = -101
vel_order[idx], idx_list[vel_order[idx]], vel[vel_order[idx]]

# %%
# ! resampling

def resample(track, n=10):
	track = track.copy()
	data = track._track.copy()
	return readtrack.TrackLike(data[::n])

def fixed_process(data, xlim=(-0.10, 0.10), delta=0.005, remove_zeros=True, remove_pairs=False):
	data = data[np.isfinite(data)]
	if remove_zeros:
		data = data[data!=0]
	b = data.size
	if remove_pairs:
		data = emanalyse.remove_pairs(data ,delta=delta)
	c = data.size
	print('remove pairs fraction', (b - c)/data.size)
	
	keep = np.logical_and(data > xlim[0], data < xlim[1])
	return data[keep]

track = track_list[654]
track = track_list[vel_order[-10]]
track = track_list[vel_order[-101]]
# track = track_list[vel_order[200]]
# track = track_list[vel_order[100]]
# track = track_list[vel_order[10]]

shstyle = dict(element="step", fill=False, alpha=0.6)
vstyle = dict(lw=2, alpha=0.4, c='k', linestyle='--')
with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots(figsize=(4,4))

	ax.axvline(0, **vstyle)
	ax.axvline(0.03, **vstyle)

	for n_sample in [1,2,5,10]:
		udata, wavemodel, lptr, meta = get_data(resample(track, n_sample))
		data = fixed_process(udata, xlim=(-0.10, np.sqrt(n_sample)*0.10), remove_zeros=True, remove_pairs=True)
		# ax.set_xlim(-0.1,0.2)
		ax.set_xlim(-0.1,0.1)
		print(len(data))


		sns.histplot(data, ax=ax, **shstyle)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))

# %%

# shape
lam = 107
# shape
b = 1/40
# rate
c = 2.5
# TODO
# c needs to be a fixed value but there are only a handful of reasonable values ...

# lam, b = 1/b, 1/lam

eml = emanalyse.EML(0, 1/b, lam)

lgm = emanalyse.LGM(lam, b, c)
# lgm1 = LGM(lam, b, 1)

xlim = 0.2 * np.array([-1, 1])
# xlim = 1.0 * np.array([-1, 1])
x1 = np.linspace(xlim[0], xlim[1], 1000)
fig, ax = plt.subplots(figsize=(6,4))
ax.axvline(0, **vstyle)

h1, = ax.plot(x1, lgm.pdf(x1))

h2, = ax.plot(x1, eml.pdf(x1), linestyle='--')
ax.legend([h1, h2], ['lgm', 'eml'])

print('int', scipy.integrate.simpson(lgm.pdf(x1), x1))
print('int', scipy.integrate.simpson(eml.pdf(x1), x1))

# %%

fig, ax = plt.subplots(figsize=(6,4))
ax.axvline(0, **vstyle)
# for c in [1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
for c in [2, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0]:
	lgm = emanalyse.LGM(lam, b, c, force_real=True)

	xlim = 0.2 * np.array([-1, 1])
	x1 = np.linspace(xlim[0], xlim[1], 1000)

	h1, = ax.plot(x1, lgm.pdf(x1))

# %%

# gammainc(1.5, 1)
-0.1**(-1.5)

c = 1.5
emanalyse.LGM(lam, b, c).pdf(np.array([-0.01, 0, 0.01]))

# %%

c = 3
def L_LGM(par=[107, 1/40, c]):
	construct = emanalyse.MixtureModel(numerical=True, n_constraints=1, err='laplace')
	construct.add_impulse(0, fix_loc=True)
	construct.add_gamma(par, err='laplace', force_real=False)
	def pack(mix):
		return np.array([1/mix[1].tau, mix[1].b])
	def unpack(self, x):
		lam, b = x
		self.mix[0].scale = 1/lam
		self.mix[1].tau = 1/lam
		self.mix[1].b = b
		# self.mix[1].c = c
	construct.set_numerical_instructions(pack, unpack)
	delta = 1e-6
	# construct.set_bounds([[delta, None], [delta, None], [delta, None]])
	rate_bound = [1, 10]	
	construct.set_bounds([[delta, None], [delta, None]])
	return construct

# construct.mixed_integer = True

n_sample = 1

track = track_list[vel_order[-101]]
udata, wavemodel, lptr, meta = get_data(resample(track, n_sample))
_xlim = (-0.10, np.sqrt(n_sample)*0.10)
data = fixed_process(udata, xlim=_xlim, remove_zeros=True)

# %%
# ! work

n_maxiter = 100
c_range = range(1, 10, 1)
# c_range = range(1, 2, 1)

with support.Timer():

	result_list = []

	result_fun = np.inf

	for c in c_range:
		print('c = ', c)
		par=[107, 1/40, c]
		construct = L_LGM(par)
		construct.fit(data, tol=1e-5, maxiter=n_maxiter)
		# 
		result_list.append([c, construct.result, construct])
		print(result_fun , construct.result.fun)
		if construct.result.fun < result_fun:
			result_fun = construct.result.fun
		else:
			break
		print('iterations', construct.n_iteration)

result_list

# %%
print(len(result_list))
sort_result = sorted([(c, r.fun, constr) for (c, r, constr) in result_list], key= lambda t: t[1])

# %%

def ks_test(rvs, construct, _xlim=None):
	if _xlim is None:
		xlim = rvs.min(), rvs.max()
	else:
		xlim = _xlim
	xspace = np.linspace(xlim[0], xlim[-1], 20000)
	pdf = construct.pdf(xspace)
	cdf = scipy.integrate.cumulative_trapezoid(pdf, xspace)
	_xspace = (xspace[1:]+xspace[:-1])/2
	f_cdf = scipy.interpolate.interp1d(_xspace, cdf, fill_value=(0,1), bounds_error=False)
	res = scipy.stats.kstest(rvs, f_cdf)
	return res.statistic, res.pvalue


ks_stat_list = [ks_test(data, _construct)[0] for _, _, _construct in result_list]
print('ks_stats', ks_stat_list)

# %%
_c, _r, constr = result_list[np.argmin(ks_stat_list)]
# _c, _r, constr = sort_result[0]
ks_stat, pv = ks_test(data, constr, _xlim=None)
print('ks_stat', ks_stat)


print('weights', construct.weights)

with mpl.rc_context({"font.size": 18}):
	fig, ax = plt.subplots(figsize = (6,4))
	emanalyse.pub_context_plot(ax, data, constr, xlim=(-0.1, 0.2))
	ax.set_ylim((0,24))

	print('lambda', 1/constr.mix[1].tau)

	def annotate(ax, c, w0, b):
		_x = 0.6
		ax.text(_x, 0.8, 'c = {}'.format(c), transform=ax.transAxes, fontsize=24)
		ax.text(_x, 0.6, r'$w_0 = {:.2f}$'.format(w0), transform=ax.transAxes, fontsize=24)
		ax.text(_x, 0.4, r'$1/b = {:.1f}$'.format(1/b), transform=ax.transAxes, fontsize=24)
	annotate(ax, _c, constr.weights[0], constr.mix[1].b)


if publish:
	pub.save_figure("example_101_n_01")

# %%
# switch to publication figures
pub.set_write_dir("/home/dan/usb_twitching/sparseml/EM_paper/prx/")
print("writing figures to", pub.writedir)

# %%
# !test
from IPython.display import display, HTML
def listfm(l, n=5):
	return '[' + ','.join([('{:.%df}' % n).format(x) for x in l]) + ']'
def fm(f, n=3):
	return ('{:.%df}' % n).format(f)

formatter = {
	'parameters' : listfm,
	'weights' : listfm
}
def gmdisplay(df):
	display(HTML(df.to_html(formatters=formatter, float_format=fm)))

# df = pd.read_pickle(join(pili.root, 'notebook/em_algorithm/mdf.pkl'))

mm_sample = 10
sample_dir = 'notebook/em_algorithm/m01_sample_{:02d}'.format(mm_sample)
df = pd.read_pickle(join(pili.root, sample_dir, 'df_0675.pkl'))
gmdisplay(df)

class Mixture(object):

	def __init__(self, mix, weights):
		self.mix = mix
		self.weights = weights

	def pdf(self, x):
		res = np.zeros_like(x)
		for w, mix in zip(self.weights, self.mix):
			res += w * mix.pdf(x)
		return res

def reconstruct_llgm(em):
	m0, scale, lam, b, c = em['parameters']
	w1, w2 = em['weights']
	mix = [emanalyse.Laplace(m0, scale), emanalyse.LGM(lam, b, c)]
	construct = Mixture(mix, em['weights'])
	return construct


m_idx = df['bic'].argmin()
em = df.iloc[m_idx]
print('best c ', em['c'])
mm = reconstruct_llgm(em)


def get_mm_data(track, n_sample=1, remove_zeros=True, remove_pairs=True):
	udata, wavemodel, lptr, meta = get_data(resample(track, n_sample))
	mm_data = fixed_process(udata, xlim=(-0.10, np.sqrt(n_sample)*0.10), remove_zeros=remove_zeros, remove_pairs=remove_pairs)
	return mm_data

# mm_track = track_list[650]
mm_track = track_list[vel_order[-101]]
mm_data = get_mm_data(mm_track, n_sample=mm_sample)

with mpl.rc_context({"font.size": 18}):
	fig, ax = plt.subplots(figsize = (6,4))
	emanalyse.pub_context_plot(ax, mm_data, mm, xlim=(-0.08, 0.16))


# %%
# %%
from glob import glob
# load all the data sets
sampling_numbers = [1, 2, 5, 10]
dflist_list = []
for n_sample in sampling_numbers:
	sample_dir = join(pili.root, 'notebook/em_algorithm/m01_sample_{:02d}'.format(n_sample))
	list_dir = sorted(glob(join(sample_dir, 'df_*.pkl')))
	_dflist = [pd.read_pickle(path) for path in list_dir]
	dflist_list.append(_dflist)

gmdisplay(dflist_list[0][0])

# %%
# ! where to start...
# * Example plots

def best_bic(df):
	m_idx = df['bic'].argmin()
	return df.iloc[m_idx]

def best_ks(df):
	m_idx = df['ks_stat'].argmin()
	return df.iloc[m_idx]


index = vel_order[-101]
sampling_df = [dflist_list[i][index] for i in range(len(sampling_numbers))]
em_list = [best_bic(df) for df in sampling_df]
# em_list = [best_ks(df) for df in sampling_df]

# ! can't load m01_sample_01 because its not finished calculating!
for i in range(1,4):
# for i in range(0,1):
	mm = reconstruct_llgm(em_list[i])
	mm_data = get_mm_data(track_list[index], sampling_numbers[i], remove_zeros=True)
	print('n', sampling_numbers[i], 'c', mm.mix[1].c)
	print('weights', mm.weights)
	print('lambda', 1/mm.mix[1].tau)
	print()

	with mpl.rc_context({"font.size": 18}):
		fig, ax = plt.subplots(figsize = (6,4))
		emanalyse.pub_context_plot(ax, mm_data, mm, xlim=(-0.08, 0.16))
		annotate(ax, mm.mix[1].c, mm.weights[0], mm.mix[1].b)

	if i == 0:
		pass
	else:
		ax.set_ylim((0,30))

	publish = False
	if publish:
		pub.save_figure("example_101_n_{:02d}".format(sampling_numbers[i]))
	

# %%
# ! another trajectory
index = vel_order[101]
print('vel', vel_order[index], idx_list[vel_order[index]], vel[vel_order[index]])
# ! sampling 01 not finished
sampling_df = [dflist_list[i][index] for i in range(1, len(sampling_numbers))]
# em_list = [best_bic(df) for df in sampling_df]
em_list = [best_ks(df) for df in sampling_df]

for i, n in enumerate(sampling_numbers[1:]):
	mm = reconstruct_llgm(em_list[i])
	mm_data = get_mm_data(track_list[index], n, remove_zeros=True)
	print('n', n, 'c', mm.mix[1].c)
	print('weights', mm.weights)
	print('lambda', 1/mm.mix[1].tau)
	print()

	with mpl.rc_context({"font.size": 18}):
		fig, ax = plt.subplots(figsize = (6,4))
		ax.axvline(0.03, **vstyle)
		emanalyse.pub_context_plot(ax, mm_data, mm, xlim=(-0.08, 0.10))
		annotate(ax, mm.mix[1].c, mm.weights[0], mm.mix[1].b)

	ax.set_ylim((0,60))

	publish = True
	if publish:
		pub.save_figure("example_+101_n_{:02d}".format(n))
	
# %%
vel[vel_order[101]]

# %%
# ! yet another trajectory
index = 100
print('vel', vel_order[index], idx_list[vel_order[index]], vel[vel_order[index]])
# ! sampling 01 not finished
sampling_df = [dflist_list[i][index] for i in range(1, len(sampling_numbers))]
# em_list = [best_bic(df) for df in sampling_df]
em_list = [best_ks(df) for df in sampling_df]

for i, n in enumerate(sampling_numbers[1:]):
	mm = reconstruct_llgm(em_list[i])
	mm_data = get_mm_data(track_list[index], n, remove_zeros=True)
	print('n', n, 'c', mm.mix[1].c)
	print('weights', mm.weights)
	print('lambda', 1/mm.mix[1].tau)
	print()

	with mpl.rc_context({"font.size": 18}):
		fig, ax = plt.subplots(figsize = (6,4))
		ax.axvline(0.03, **vstyle)
		emanalyse.pub_context_plot(ax, mm_data, mm, xlim=(-0.08, 0.10))
		annotate(ax, mm.mix[1].c, mm.weights[0], mm.mix[1].b)

	ax.set_title('')
	ax.set_ylim((0,60))

	if publish:
		pub.save_figure("example_+100_n_{:02d}".format(n))
	
# %%
# # ! yet yet another trajectory
# index = 20
# print('vel', vel_order[index], idx_list[vel_order[index]], vel[vel_order[index]])
# # ! sampling 01 not finished
# sampling_df = [dflist_list[i][index] for i in range(1, len(sampling_numbers))]
# # em_list = [best_bic(df) for df in sampling_df]
# em_list = [best_ks(df) for df in sampling_df]

# for i, n in enumerate(sampling_numbers[1:]):
# 	mm = reconstruct_llgm(em_list[i])
# 	mm_data = get_mm_data(track_list[index], n, remove_zeros=True)
# 	print('n', n, 'c', mm.mix[1].c)
# 	print('weights', mm.weights)
# 	print('lambda', 1/mm.mix[1].tau)
# 	print()

# 	with mpl.rc_context({"font.size": 18}):
# 		fig, ax = plt.subplots(figsize = (6,4))
# 		ax.axvline(0.03, **vstyle)
# 		emanalyse.pub_context_plot(ax, mm_data, mm, xlim=(-0.08, 0.10))
# 		annotate(ax, mm.mix[1].c, mm.weights[0], mm.mix[1].b)

# 	ax.set_ylim((0,60))

# 	if publish:
# 		pub.save_figure("example_+100_n_{:02d}".format(n))
	
# %%
# ! Population multiscale analysis
publish = False

# get the best estimate of the weights
best_df = [[best_bic(dflist_list[i][index]) for index in range(779)] for i in range(0,4)]
# best_df = [[best_ks(dflist_list[i][index]) for index in range(779)] for i in range(0,4)]

z = 0
npar = [(n, [em['parameters'] for em in best_df[i]]) for i, n in enumerate(sampling_numbers[z:])]
nweight = [(n, [em['weights'] for em in best_df[i]]) for i, n in enumerate(sampling_numbers[z:])]

# %%
nks_stat = [(n, np.array([em["ks_stat"] for em in best_df[i]])) for i, n in enumerate(sampling_numbers[1:])]

np.median(nks_stat[0][1]), np.median(nks_stat[1][1]), np.median(nks_stat[2][1])


# %%
from matplotlib.lines import Line2D
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']

with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots(figsize=(4,4))

	handles = []
	for i, (n, par) in enumerate(npar):
		n, weight = nweight[i]
		m0, scale, lam, b, c =  list(zip(*par))
		w0, w1 = list(zip(*weight))
		print(1, n, np.mean(w0))
		
		scatterstyle = dict(alpha=0.2)
		h = ax.scatter(vel, w0, **scatterstyle)
		h = Line2D([0], [0], marker='o', color='w', markerfacecolor=defcolor[i], markersize=13),
		handles.append(h)
		ax.set_ylim((0,1))
		ax.set_xlim((0,0.12))

	ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	ax.yaxis.set_major_locator(plt.MaxNLocator(5))

	ax.set_ylabel("$w_0$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.26, .4)
	ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)

	labels = [r'$\Delta t = 0.1$', r'$\Delta t = 0.2$', r'$\Delta t = 0.5$', r'$\Delta t = 1.0$']
	ax.legend(handles, labels, loc=(1.0,.04))

if publish:
	pub.save_figure("population_multiscale_w0")	

# %%


with mpl.rc_context(mplstyle):
	for i, (n, par) in enumerate(npar):
		fig, ax = plt.subplots(figsize=(4,4))
		n, weight = nweight[i]
		m0, scale, lam, b, c =  list(zip(*par))
		w0, w1 = list(zip(*weight))
		mod_c = np.array(c) + scipy.stats.norm(0, 0.15).rvs(len(c))
		
		scatterstyle = dict(alpha=0.2)
		h = ax.scatter(vel, mod_c, **scatterstyle)
		h = Line2D([0], [0], marker='o', color='w', markerfacecolor=defcolor[i], markersize=13),
		# ax.set_ylim((0,1))
		ax.set_xlim((0,0.12))

		ax.xaxis.set_major_locator(plt.MaxNLocator(3))
		# ax.yaxis.set_major_locator(plt.MaxNLocator(2))

		ax.set_ylabel("$c$", fontsize=32, rotation=np.pi/2)
		ax.yaxis.set_label_coords(-0.26, .4)
		ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)

	# labels = [r'$\Delta t = 0.2$', r'$\Delta t = 0.5$', r'$\Delta t = 1.0$']
	# ax.legend(handles, labels, loc=(1.0,.04))

# %%

with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots(figsize=(4,4))

	handles = []
	c_data = []
	b_data = []
	lam_data = []
	for i, (n, par) in enumerate(npar):
		n, weight = nweight[i]
		m0, scale, lam, b, c =  list(zip(*par))
		print('c.median', np.mean(c))
		c_data.append(c)
		b_data.append(1/np.array(b))
		lam_data.append(lam)

	# ax.violinplot(c_data, widths=0.8)
	# ax.set_ylim((None,6))

	# ax.violinplot(b_data)
	ax.violinplot(lam_data)

	# ax.set_ylim((0,1))
	# ax.set_xlim((0,0.12))

	# ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	# ax.yaxis.set_major_locator(plt.MaxNLocator(2))

	ax.set_ylabel("$\lambda$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.26, .4)
	ax.set_xlabel(r"$\Delta t \, (s)$", fontsize=20)
	ax.xaxis.set_ticks([1,2,3,4], labels=['0.1', '0.2', '0.5', '1.0'])

	# labels = [r'$\Delta t = 0.2$', r'$\Delta t = 0.5$', r'$\Delta t = 1.0$']
	# ax.legend(handles, labels, loc=(1.0,.04))

publish = False
# if publish:
# 	pub.save_figure("population_multiscale_lambda")	


# %%

# ! resampling using a step length

track = track_list[654]
# track = track_list[vel_order[-10]]
# track = track_list[vel_order[-101]]
track = track_list[vel_order[200]]
# track = track_list[vel_order[101]]
# track = track_list[vel_order[10]]


d_step = 0.06
ltr = _fj.linearize(track, step_d = d_step )
dt_data = ltr.get_step_dt()
print('N', len(dt_data))

stat = 'density'
with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots(figsize=(4,4))

	sns.histplot(dt_data, ax=ax, stat=stat, log_scale=False, **shstyle)
	ax.set_xlim(0,6)
	# ax.set_xlim(0,12)
	ax.set_xlabel("time (s)")

	a, b = scipy.stats.expon.fit(dt_data)
	print(a, b)
	x1 = np.linspace(0,20,1000)
	pdf = scipy.stats.expon(a,b).pdf(x1)
	ax.plot(x1, pdf)
	

scipy.stats.anderson(dt_data, dist='expon')

