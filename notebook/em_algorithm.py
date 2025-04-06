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
# implement the expectation maximisation algorithm and compare with sklearn GMM code where appropriate

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
pub.set_write_dir("/home/dan/usb_twitching/sparseml/EM_paper/prx")
print("writing figures to", pub.writedir)

import thesis.publication as thesis

import pili
from pili import support
import emanalyse
from emanalyse import pub_context_plot
import _fj
import mdl
import pwlpartition

import fjanalysis
import pwlstats

from skimage.restoration import denoise_wavelet
from skimage.restoration import  estimate_sigma

import sklearn.mixture

# %% 
mplstyle = {"font.size": 20}
vstyle = dict(alpha=0.2,c='black',linestyle='--')
notename = "em_algorithm"
publish = True
# work = False



column_width = 3.385
f_size = (column_width, column_width)
fh_size = 1.0 * np.array([column_width/2, column_width/2])
use_size = fh_size
# use_size = (4,4)

texstyle = {
	"font.size": 10, "ytick.labelsize": 9, "xtick.labelsize": 9, 
	"text.usetex": True, "axes.labelsize" : 10, "legend.frameon" : False,
	"xtick.direction": 'in', "ytick.direction": 'in'
	} 
usestyle = texstyle

usestyle = thesis.texstyle
use_size = (4,4)

# %% 
# target = join(pwlstats.root, "run/partition/candidate/no_heuristic/_candidate_pwl/")

# TEST
target = "/home/dan/usb_twitching/sparseml/run/cluster/no_heuristic/top/_top_2368/"
target = '/home/dan/usb_twitching/sparseml/run/cluster/no_heuristic/top/_top_1924'

solver = pwlstats.load_solver_at(target)

solver.partition.cache_local_coord = np.empty(solver.partition.N) # hotfix
solver.partition.update_residuals()

curve_coord = solver.partition.get_curve_coord()
udata = np.diff(curve_coord) 

# %% 
shstyle = dict(element="step", fill=False, alpha=0.8)
# xlim for plotting
xlim = (-0.08, 0.16)
# xlim for preprocess 
pxlim = (-0.16, 0.14)

# %%

# auxdir = join('notebook/em_algorithm/crawling_em_quad_2/', 'aux')
auxdir = join(pili.root, 'notebook/em_algorithm/crawling_em_quad_3/', 'aux')
# auxdir = join(pili.root, 'notebook/em_algorithm/', 'aux')
# auxdir = join(pili.root, 'notebook/em_algorithm/crawling_em_db4/', 'aux')
# auxdir = join(pili.root, 'notebook/em_algorithm/crawling_em_quad_4/', 'aux')
# auxdir = join(pili.root, 'notebook/em_algorithm/crawling_em_db4rz/', 'aux')
# auxdir = join(pili.root, 'notebook/em_algorithm/crawling_em_db4rz_fixerr/', 'aux')

with open(join(pili.root, auxdir, 'crawling_wavelist.pkl'), 'rb') as f:
	wavelist = pickle.load(f)

with open(join(pili.root, auxdir, 'crawling_lptrlist.pkl'), 'rb') as f:
	lptrlist = pickle.load(f)

with open(join(pili.root, auxdir, 'crawling_metalist.pkl'), 'rb') as f:
	metalist = pickle.load(f)

with open(join(pili.root, auxdir, 'crawling_udatalist.pkl'), 'rb') as f:
	udatalist = pickle.load(f)

# ! later we will reload the datalist from the aux directory

# important meta parameter controlling the data preprocessing
def preprocess(data, q=0.005, delta=0.005):
	data = data[np.isfinite(data)]
	data = emanalyse.remove_pairs(data ,delta=delta)
	data = emanalyse.symprocess(data, q, remove_zeros=True)
	return data

q = 0.02
delta = 0.005
datalist = [preprocess(udata, q, delta) for udata in udatalist]

# %%
# ! check the current preprocessing

def fixed_process(data, xlim=(-0.05, 0.10), delta=0.005):
	data = data[np.isfinite(data)]
	
	data = emanalyse.remove_pairs(data ,delta=delta)
	rs = data.size
	keep = np.logical_and(data > xlim[0], data < xlim[1])
	return data[keep]

u_size = np.array([udata.size for udata in udatalist])
z_size = []
r_size = []
k_size = []

for udata in udatalist:
	xlim = (-0.10, 0.10)
	data = udata[udata!=0]
	z_size.append(data.size)
	data = emanalyse.remove_pairs(data ,delta=0.005)
	r_size.append(data.size)
	keep = np.logical_and(data > xlim[0], data < xlim[1])
	data = data[keep]
	k_size.append(data.size)

r_size = np.array(r_size)
k_size = np.array(k_size)

# so the fraction removed 
print('zeros', np.median( (u_size-z_size)/u_size ))
print('+/-', np.median( (z_size-r_size)/u_size ))
print('clip', np.median( (r_size-k_size)/u_size ))

# %%
# the fraction of zeros for the example trajectory is about 7%
((u_size-z_size)/u_size)[-12]

# %%
# np.random.
# means_init = np.array([0.0, 0.03])[:, np.newaxis]
means_init = np.array([-0.03, 0.0, 0.03])[:, np.newaxis]

setting = dict(max_iter=1000, tol=1e-8)
gmm = sklearn.mixture.GaussianMixture(n_components=len(means_init), means_init=means_init, random_state=1, **setting)

data = datalist[0]
data = data[np.isfinite(data)]
_data = data.reshape(-1, 1)
gm = gmm.fit(_data)

def describe(gm):
	print('means', gm.means_.ravel())
	print('std', np.sqrt(gm.covariances_))
	print('weights', gm.weights_)
	print('bic', gm.bic(_data))

describe(gm)

labels = gm.predict(_data)

sstyle = dict(element="step", fill=False, alpha=0.8, stat='density')
def sample_plot(ax, gm, data, n_sample=1000):
	sample, sample_label = gm.sample(n_sample)
	sample = sample.ravel()

	sns.histplot(sample, ax=ax, **sstyle)
	sns.histplot(data, ax=ax, **sstyle)
	ax.set_xlim(xlim)

	for mean in gm.means_.ravel():
		ax.axvline(mean, **vstyle)

	ax.legend(['mixture', 'data'], loc='upper right')

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
def quick_plot(ax, gm, data, separate=True):
	lstyle = dict(alpha=0.6)

	xspace = np.linspace(data.min(), data.max(), 1000)

	if separate:
		color = itertools.cycle([defcolor[0]])
	else:
		color = itertools.cycle(defcolor)

	it = zip(gm.means_.ravel(), gm.covariances_.ravel(), gm.weights_.ravel())
	for mean, var, weight in it:
		mm = weight * scipy.stats.norm(mean, np.sqrt(var)).pdf(xspace)
		h1,  = ax.plot(xspace, mm, color=next(color), **lstyle)

	sstyle = dict(element="step", fill=False, alpha=0.8, stat='density')
	sns.histplot(data, ax=ax, linewidth=2, color=defcolor[1], **sstyle)
	h2 = ax.lines[-1]
	ax.set_xlim(xlim)
	ax.legend([h1, h2], ["prediction", "data"], fontsize=18, loc='upper right')

	for mean in gm.means_.ravel():
		ax.axvline(mean, **vstyle)

	# ax.legend(['mixture', 'data'], loc='upper right')


with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots(figsize=(6,4))
	quick_plot(ax, gm, data)

# %% 
# * INVESTIGATE the effect of preprocessing threshold
# first process out the zeroes

# def fit(data, means_init):
# 	gmm = sklearn.mixture.GaussianMixture(n_components=len(means_init), means_init=means_init, **setting)
# 	gm = gmm.fit(data.reshape(-1, 1))
# 	return gm

# quant = [0, 0.02, 0.03, 0.04, 0.05, 0.06]
# gmlist = []
# pdata = []
# for q in quant:
# 	p_data = emanalyse.asymprocess(udata, q, side='right')

# 	means_init = np.array([-0.03, 0.0, 0.03])[:, np.newaxis]
# 	gm = fit(p_data, means_init)
# 	gmlist.append(gm)
# 	pdata.append(p_data)


# %%

# why is this so slow?
# for gm, p_data in zip(gmlist, pdata):


# 	describe(gm)
# 	print()
# 	fused = (p_data.size/udata.size)

# 	with mpl.rc_context(mplstyle):
# 		fig, ax = plt.subplots(figsize=(6,4))
# 		quick_plot(ax, gm, p_data)
# 		# ax.set_title("[{:.3f}, {:.3f}] ({:4.1f}%)".format(_pxlim[0], _pxlim[1], 100*fused))
# 		ax.set_title("({:4.1f}%)".format(100*fused))


# %%
# * RUN sklearn GMM on the whole dataset

# gmlist2 = []
# gmlist3 = []

# setting = dict(max_iter=1000, tol=1e-8)

# for i, data in enumerate(datalist):
# # for i, data in list(enumerate(datalist))[:5]:
# 	if i % 10 == 0:
# 		print('{}/{}'.format(i, len(datalist)))
# 	means_init = np.array([0.0, 0.03])[:, np.newaxis]
# 	gmm = sklearn.mixture.GaussianMixture(n_components=len(means_init), means_init=means_init, **setting)
# 	_data.reshape(-1, 1)
# 	gm = gmm.fit(_data)
# 	# print(gm.means_)
# 	gmlist2.append(gm)

# 	means_init3 = np.array([0.0, 0.03, -0.03])[:, np.newaxis]
# 	gmm = sklearn.mixture.GaussianMixture(n_components=len(means_init3), means_init=means_init3, **setting)
# 	gm = gmm.fit(_data)

# 	gmlist3.append(gm)

# # %%

# index = 0
# gm = gmlist3[index]
# print('size', datalist[index].size)


# with mpl.rc_context(mplstyle):
# 	fig, ax = plt.subplots(figsize=(6,4))
# 	quick_plot(ax, gm, datalist[index])

# # %%
# def get_row(gm, data):
# 	local = {}
# 	local["means"] = gm.means_.ravel()
# 	local["std"] = np.sqrt(gm.covariances_)
# 	local['weights'] = gm.weights_
# 	local['bic'] = gm.bic(data)
# 	local['score'] = gm.score(data)
# 	return local

# def concat(rowlist):
# 	data = {}
# 	for key in rowlist[0]:
# 		data[key] = []
# 	for row in rowlist:
# 		for key in row:
# 			data[key].append(row[key])
# 	return data

# gmdf2 = pd.DataFrame(concat([get_row(gm, data.reshape(-1,1)) for gm, data in zip(gmlist2, datalist)]))
# gmdf3 = pd.DataFrame(concat([get_row(gm, data.reshape(-1,1)) for gm, data in zip(gmlist3, datalist)]))


# [(data.reshape(-1, 1)) for gm, data in zip(gmlist, datalist)]
# [gm.bic(data.reshape(-1, 1)) for gm, data in zip(gmlist, datalist)]


# %%
# analysing GMM2 vs GMM3 model

# best = (gmdf2['bic'] < gmdf3['bic']).to_numpy()
# gmdf2['best'] = best

# # sns.histplot(data=gmdf2, x)
# fig, ax = plt.subplots(figsize=(6,4))
# ax.violinplot([vel[best], vel[~best]])
# ax.legend(['gmm2', 'gmm3'])
# ax.set_ylabel('mean velocity')


# fig, ax = plt.subplots(figsize=(6,4))
# m0, m1 = list(zip(*gmdf2['means']))
# print(m1)
# sns.histplot(m1)
# ax.set_xlim(0, 0.03)
# # ax.violinplot([m1])
# # ax.set_ylim(0, 0.05)


# %%

# equivalent of emanalyse.quick_plot for my MM implementation
def context_plot(ax, mm, data, xlim=xlim, separate=True):
	defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
	lstyle = dict(alpha=0.6)

	xspace = np.linspace(data.min(), data.max(), 1000)

	color = itertools.cycle([defcolor[0]])

	if separate:
		for i, mix in enumerate(mm.mix):
			h1, = ax.plot(xspace, mm.weights[i] * mix.pdf(xspace), color=next(color), **lstyle) 
	else:
		pdf = np.stack([mm.weights[i] * mix.pdf(xspace) for i, mix in enumerate(mm.mix)]).sum(axis=0)
		h1, = ax.plot(xspace, pdf, **lstyle)

	sns.histplot(data, ax=ax, linewidth=2, color=defcolor[1], **sstyle)

	h2 = ax.lines[-1]
	ax.set_xlim(xlim)
	ax.legend([h1, h2], ["prediction", "data"], fontsize=18, loc='upper right')

	ax.axvline(0, **vstyle)
	ax.axvline(-0.03, **vstyle)
	ax.axvline(0.03, **vstyle)



# %%
# test EMG
from emanalyse import EMG, Gauss, MixtureModel, EML, NTN

sigma = 0.009
lam = 40

mix = EMG(0, sigma, lam)
xlim = (-0.05, 0.10)
xspace = np.linspace(xlim[0], xlim[-1], 1000)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(xspace, mix.pdf(xspace))
ax.axvline(0, **vstyle)
ax.axvline(0 + 1/lam, **vstyle)


# %%
# test exponentially modified laplace distribution

x1 = np.linspace(xlim[0], xlim[-1], 1000)
eml = emanalyse.EML(0.0, 40, 120)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x1, eml.pdf(x1))
ax.axvline(0, **vstyle)
ax.set_title("exponentially modified laplace distribution")

# check this is a pdf
# scipy.integrate.simpson(eml.pdf(x2), x2)

# test this by computing an explicit convolution numerically
# p1 = scipy.stats.expon(0, 1/40).pdf(xspace)
# p2 = scipy.stats.laplace(0, 1/120).pdf(x2)
# fig, ax = plt.subplots(figsize=(6,4))
# conv = np.convolve(p1, p2, mode='full')
# # conv /= scipy.integrate.simpson(conv, xspace)
# ax.plot(conv)


# %%
# test N-TN (convolution of Normal and Truncated Normal distributions)

x1 = np.linspace(xlim[0], xlim[-1], 1000)
ntn = NTN(0.00, 0.009, 0.009)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x1, ntn.pdf(x1))
ax.axvline(0, **vstyle)
ax.set_title("N-TN distribution")

# check normalisation
scipy.integrate.simpson(ntn.pdf(x1), x1)

# %%
# implement the convolution of laplace and truncated normal distribution

from emanalyse import LTN

ltn = LTN(0.00, 0.009, 40)

x1 = np.linspace(xlim[0], xlim[-1], 1000)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x1, ltn.pdf(x1))
ax.axvline(0, **vstyle)
ax.set_title("L-TN distribution")

x1 = np.linspace(10*xlim[0], 10*xlim[-1], 1000)
scipy.integrate.simpson(ltn.pdf(x1), x1)

# %%

# defaults
_sigma = 0.009
_lam= 40.0

# define models
def gmm2(sigma=_sigma):
	construct = MixtureModel(sigma)
	construct.add_impulse(0, fix_loc=True)
	construct.add_impulse(0.03)
	return construct

def gmm2_constraint(sigma=_sigma):
	construct = MixtureModel(sigma, numerical=True, n_constraints=1, sigma_constraint=True)
	construct.add_impulse(0, fix_loc=True)
	construct.add_impulse(0.03)
	def pack(mix):
		return np.array([mix[0].scale, mix[1].loc])
	def unpack(self, x):
		sigma, m1 = x
		self.mix[0].scale = sigma
		self.mix[1].scale = sigma
		self.mix[1].loc = m1
	construct.set_numerical_instructions(pack, unpack)
	return construct

def gmm3(sigma=_sigma):
	construct = MixtureModel(sigma)
	construct.add_impulse(0)
	construct.add_impulse(0.03)
	construct.add_impulse(-0.03)
	return construct
	
def g_emg_constraint(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, numerical=True, n_constraints=1, sigma_constraint=True)
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0, lam)
	def pack(mix):
		return np.array([mix[0].scale, mix[1].loc, mix[1].lam])
	def unpack(self, x):
		sigma, m1, lam = x
		self.mix[0].scale = sigma
		self.mix[1].sigma =  sigma
		self.mix[1].loc =  m1
		self.mix[1].lam =  lam
	construct.set_numerical_instructions(pack, unpack)
	return construct

def g_emg_fixed(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, numerical=True, n_constraints=1, sigma_constraint=True)
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0, lam, fix_loc=True)
	def pack(mix):
		return np.array([mix[0].scale, mix[1].lam])
	def unpack(self, x):
		sigma, lam = x
		self.mix[0].scale = sigma
		self.mix[1].sigma =  sigma
		self.mix[1].lam =  lam
	construct.set_numerical_instructions(pack, unpack)
	return construct

def g_emg(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, numerical=True, n_constraints=0, sigma_constraint=False)
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0, lam)
	def pack(mix):
		return np.array([mix[0].scale, mix[1].loc, mix[1].sigma, mix[1].lam])
	def unpack(self, x):
		sigma, m1, emgsigma, lam = x
		self.mix[1].sigma =  emgsigma
		self.mix[0].scale = sigma
		self.mix[1].loc =  m1
		self.mix[1].lam =  lam
	construct.set_numerical_instructions(pack, unpack)
	return construct

# a.k.a L_eml_constraint
def L_eml(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, 
		numerical=True, n_constraints=1, sigma_constraint=False, err='laplace')
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0.01, err='laplace')
	def pack(mix):
		# l1 is the exponential parameter
		# l2 is the error parameter
		return np.array([mix[1].l1, mix[1].loc, mix[1].l2])
	def unpack(self, x):
		l1, m1, l2 = x
		self.mix[0].scale = 1/l2
		self.mix[1].loc = m1
		self.mix[1].l1 = l1
		self.mix[1].l2 = l2
	construct.set_numerical_instructions(pack, unpack)
	l1_max = 500
	construct.set_bounds([[0, l1_max], [0, None], [0, None]])
	return construct


def L_eml_fixed(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, numerical=True, n_constraints=1, sigma_constraint=False, err='laplace')
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0.0, None, err='laplace', fix_loc=True)
	def pack(mix):
		return np.array([mix[1].l1, mix[1].l2])
	def unpack(self, x):
		l1, l2 = x
		self.mix[0].scale = 1/l2
		self.mix[1].l1 = l1
		self.mix[1].l2 = l2
	construct.set_numerical_instructions(pack, unpack)
	return construct

# Laplace + Normal-Laplace
def L_NL(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, 
		numerical=True, n_constraints=1, sigma_constraint=False, err='laplace')
	construct.add_impulse(0)
	construct.add_normal(0.02, sigma, lam, err='laplace')
	def pack(mix):
		return np.array([mix[1].lam, mix[1].loc, mix[1].sigma])
	def unpack(self, x):
		lam, m1, sigma = x
		self.mix[0].scale = 1/lam
		self.mix[1].loc = m1
		self.mix[1].sigma = sigma
		self.mix[1].lam = lam
	construct.set_numerical_instructions(pack, unpack)
	return construct

# 
def L_NL_m0(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, 
		numerical=True, n_constraints=1, sigma_constraint=False, err='laplace')
	construct.add_impulse(0)
	construct.add_normal(0.02, sigma, lam, err='laplace')
	def pack(mix):
		return np.array([mix[0].loc, mix[1].lam, mix[1].loc, mix[1].sigma])
	def unpack(self, x):
		m0, lam, m1, sigma = x
		self.mix[0].loc = m0
		self.mix[0].scale = 1/lam
		self.mix[1].loc = m1
		self.mix[1].sigma = sigma
		self.mix[1].lam = lam
	construct.set_numerical_instructions(pack, unpack)
	return construct

def N_NTN(par=[0.00, 0.09, 0.009]):
	sigma = par[1]
	construct = MixtureModel(sigma,
		numerical=True, n_constraints=1, err='normal')
	construct.add_impulse(0, fix_loc=True)
	construct.add_truncated_normal(par , err='normal')
	def pack(mix):
		return np.array([mix[1].loc, mix[1].s, mix[1].sigma])
	def unpack(self, x):
		m1, s, sigma = x
		self.mix[0].scale = s
		self.mix[1].loc = m1
		self.mix[1].s = s
		self.mix[1].sigma = sigma
	construct.set_numerical_instructions(pack, unpack)
	delta = 1e-6
	construct.set_bounds([[0, None], [delta, None], [delta, None]])
	return construct

def L_LTN(par=[0.00, 0.009, 1/0.013]):
	construct = MixtureModel(sigma, 
		numerical=True, n_constraints=1, err='laplace')
	construct.add_impulse(0, fix_loc=True)
	construct.add_truncated_normal(par , err='laplace')
	def pack(mix):
		return np.array([mix[1].loc, mix[1].sigma, mix[1].lam])
	def unpack(self, x):
		m1, sigma, lam = x
		self.mix[0].scale = 1/lam
		self.mix[1].loc = m1
		self.mix[1].sigma = sigma
		self.mix[1].lam =lam 
	construct.set_numerical_instructions(pack, unpack)
	delta = 1e-6
	construct.set_bounds([[0, None], [delta, None], [delta, None]])
	return construct

# explictly add +/- peaks 
def l_eml_pm(par=[0.00, 0.009, 1/0.013]):
	construct = MixtureModel(numerical=True, n_constraints=1)
	construct.add_impulse(0, fix_loc=True, err='laplace')
	construct.add_exponential(0.01, None, err='laplace')
	construct.add_impulse(0.03, fix_loc=True, err='normal')
	construct.add_impulse(-0.03, fix_loc=True, err='normal')
	def pack(mix):
		return np.array([mix[1].l1, mix[1].loc, mix[1].l2, mix[2].scale, mix[3].scale])
	def unpack(self, x):
		l1, m1, l2, m_scale, p_scale = x
		self.mix[0].scale = 1/l2
		self.mix[1].loc = m1
		self.mix[1].l1 = l1
		self.mix[1].l2 = l2
		self.mix[2].scale = m_scale
		self.mix[3].scale = p_scale
	construct.set_numerical_instructions(pack, unpack)
	delta = 1e-6
	construct.set_bounds([[0, None], [delta, None], [delta, None], [delta, None], [delta, None]])
	return construct


# %%
# load the metadata aswell
load_path = join(pili.root, 'notebook/thesis_classification/crawling.npy')
idx_list = np.loadtxt(load_path).astype(int)
track_list = _fj.trackload_original(idx_list)

ldata = fjanalysis.load_summary()
topdata = [ldata[i] for i in idx_list]
vel = np.array([ld['lvel']['mean'] for ld in topdata])
print(vel.size)


vel_order = np.argsort(vel)
print(vel[vel_order[-1]])

vel.min(), vel.max()

# index = 766
# %%
200/len(track_list[766])
np.median(vel)

# %%

index = 0
index = 654
index = -12
# index = 694

# index = -12,-16
index = vel_order[-16]

print('mean velocity', vel[index])
udata = udatalist[index]
udata = udata[udata!=0]
clip_x = (-0.1, 0.1)
keep = np.logical_and(udata > clip_x[0], udata < clip_x[1])
udata = udata[keep]

data = udata

# def fixed_process(data, xlim=(-0.05, 0.10), delta=0.005, remove_zeros=True):
# 	data = data[np.isfinite(data)]
# 	if remove_zeros:
# 		data = data[data!=0]
# 	data = emanalyse.remove_pairs(data ,delta=delta)
# 	keep = np.logical_and(data > xlim[0], data < xlim[1])
# 	return data[keep]

# data = fixed_process(udata)

print('data limits', data.min(), data.max())

fig, ax = plt.subplots(figsize=(6,4))
ax.axvline(-0.03, **vstyle)
ax.axvline(0.03, **vstyle)
sns.histplot(data, ax=ax, **shstyle)
# sns.histplot(udata, ax=ax, **shstyle)
ax.set_xlim((-0.08, 0.12))
# ax.set_xlim(xlim)


data.size

# %%
# test laplace error distribution model

# construct = gmm2()
# construct = g_emg(sigma)
# construct = g_emg_constraint(sigma)
construct = L_eml()
# construct = l_eml_fixed()
# construct = g_emg_fixed()
# construct = l_nl()
# construct = N_NTN([0.0, 0.009, 0.009])
# construct = l_eml_pm()
# par = [0.02, 0.009, 1/0.013]
# construct = L_LTN(par)
print('check n_parameters', construct.n_parameters(), construct.n_constraints)


# %%

with support.Timer():
	construct.fit(data, tol=1e-5, maxiter=400, weights=[0.5, 0.5, 0.01, 0.01])
	print('iterations', construct.n_iteration)

# %%


def ks_test(rvs, construct, _xlim=(-0.05, 0.1)):
	xspace = np.linspace(xlim[0], xlim[-1], 2000)
	pdf = construct.pdf(xspace)
	# scipy. integrate.simpson(pdf, xspace)
	cdf = scipy.integrate.cumulative_trapezoid(pdf, xspace)
	_xspace = (xspace[1:]+xspace[:-1])/2
	f_cdf = scipy.interpolate.interp1d(_xspace, cdf, fill_value=(0,1), bounds_error=False)

	res = scipy.stats.kstest(rvs, f_cdf)
	return res.statistic, res.pvalue

def describe_construct(construct):
	print('weights', construct.weights)
	for mix in construct.mix:
		print(mix.describe())

	print('bic', construct.bic())
	print('n parameters', construct.n_parameters())

describe_construct(construct)

# with mpl.rc_context({"font.size": 18}):
# 	fig, ax = plt.subplots(figsize = (6,4))
# 	context_plot(ax, construct, data, separate=True)

# with mpl.rc_context({"font.size": 18}):
# 	fig, ax = plt.subplots(figsize = (6,4))
# 	context_plot(ax, construct, data, separate=False)



with mpl.rc_context({"font.size": 18}):
	fig, ax = plt.subplots(figsize = (6,4))
	pub_context_plot(ax, data, construct)

	print('ks_statistic', ks_test(data, construct))




# %%

model_names = ['gmm2', 'gmm2_constraint', 'g_emg', 'g_emg_constraint', 'gmm3']
model_list = [gmm2, gmm2_constraint, g_emg, g_emg_constraint, gmm3]

model_names = ['gmm2', 'g_emg_constraint', 'g_emg_fixed', 'l_eml', 'l_eml_fixed', 'l_nl']
model_list = [gmm2, g_emg_constraint, g_emg_fixed, L_eml, L_eml_fixed, L_NL]

model_names = ['l_eml', 'N_NTN']
model_list = [L_eml, N_NTN]

model_names = []
model_list = []

# model_names = []
# model_list = []


# test all models
with support.Timer():
	models = []
	for i, model in enumerate(model_list):
		print('fitting ', model_names[i])
		mm = model()
		mm.fit(data, tol=1e-4, maxiter=400)
		models.append(mm)

# %%
# 
for i, mm in enumerate(models):
	print(model_names[i], 'bic', mm.bic(), 'n_parameters', mm.n_parameters())
	with mpl.rc_context({"font.size": 18}):
		fig, ax = plt.subplots(figsize = (6,4))
		pub_context_plot(ax, data, mm)
		ax.set_title(model_names[i])
	
		print('ks_statistic',  ks_test(data, mm))

# %%
# create a dataframe

def dataframe(models):
	data = {
		'names' : model_names,
		'n_free' : [mm.n_parameters() for mm in models],
		'n_constraints' : [mm.n_constraints for mm in models],
		'bic' : [mm.bic() for mm in models],
		'parameters' : [mm.list_parameters() for mm in models],
		'weights' : [mm.weights for mm in models],
		'n_iteration' : [mm.n_iteration for mm in models]
	}

	return pd.DataFrame(data)

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

gmdisplay(dataframe(models))


# %%

# _construct = gmm2_constraint()
_construct = gmm2()
with support.Timer():
	_construct.fit(data, tol=1e-5, maxiter=400)
print('weights', _construct.weights)
for mix in _construct.mix:
	print(mix.describe())

print('bic', _construct.bic())
_construct.n_parameters()


with mpl.rc_context({"font.size": 18}):
	fig, ax = plt.subplots(figsize = (6,4))
	context_plot(ax, _construct, data)

# %%
# how about fitting an exponential error distribution?

left = data[data<0]

a, b = scipy.stats.expon.fit(-left)
print(a,b)
xspace = np.linspace(0, -left.min())
scipy.stats.anderson(-left, dist='expon')
fit = scipy.stats.expon(a, b).pdf(xspace)

fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(-left, ax=ax, stat='density', **shstyle)
ax.plot(xspace, fit)

# %%
# test load a dataframe

# df = pd.read_pickle(join(pili.root, 'notebook/em_algorithm/crawling_em/df_0004.pkl'))
# gmdisplay(df)


# df = pd.read_pickle(join(pili.root, 'notebook/em_algorithm/crawling_em_aux/df_0004.pkl'))
# gmdisplay(df)


# df = pd.read_pickle(join(pili.root, 'notebook/em_algorithm/df.pkl'))
# gmdisplay(df)

# df = pd.read_pickle(join(pili.root, 'notebook/em_algorithm/crawling_em_truncated/df_0004.pkl'))
# gmdisplay(df)

df = pd.read_pickle(join(pili.root, 'notebook/em_algorithm/crawling_em_quad/df_2751.pkl'))

df = pd.read_pickle(join(pili.root, 'notebook/em_algorithm/crawling_em_db4rz_fixerr/df_2751.pkl'))
gmdisplay(df)



# %%
# load an entire dataset
t1 = join(pili.root, 'notebook/em_algorithm/crawling_em/', 'df_*.pkl')
t2 = join(pili.root, 'notebook/em_algorithm/crawling_em_aux/', 'df_*.pkl')
tfix = join(pili.root, 'notebook/em_algorithm/crawling_em_zfix/', 'df_*.pkl')
# t_trunc = join(pili.root, 'notebook/em_algorithm/crawling_em_truncated/', 'df_*.pkl')
q1 = 'notebook/em_algorithm/crawling_em_quad/'
quad = join(pili.root, q1, 'df_*.pkl')
q2 = 'notebook/em_algorithm/crawling_em_quad_2/'
glob_q2 = join(pili.root, q2, 'df_*.pkl')

q3 = 'notebook/em_algorithm/crawling_em_quad_3/'
glob_q3 = join(pili.root, q3, 'df_*.pkl')

q4 = 'notebook/em_algorithm/crawling_em_quad_3/'
glob_q4 = join(pili.root, q4, 'df_*.pkl')

# qdb4 = 'notebook/em_algorithm/crawling_em_db4rz/'
qdb4 = 'notebook/em_algorithm/crawling_em_db4rz_fixerr/'
glob_qdb4 = join(pili.root, qdb4, 'df_*.pkl')

from glob import glob


def concat(a, b):
	return pd.concat([a, b], ignore_index=True)

# df1 = [pd.read_pickle(at) for at in sorted(glob(t1))]
# df2 = [pd.read_pickle(at) for at in sorted(glob(t2))]
# dflist = [concat(a, b) for a, b, in zip(df1, df2)]

df1 = [pd.read_pickle(at) for at in sorted(glob(tfix))]
# df2 = [pd.read_pickle(at) for at in sorted(glob(quad))]
# df2 = [pd.read_pickle(at) for at in sorted(glob(glob_q2))]
df2 = [pd.read_pickle(at) for at in sorted(glob(glob_q3))]
# df2 = [pd.read_pickle(at) for at in sorted(glob(glob_q4))]

# df2 = [pd.read_pickle(at) for at in sorted(glob(glob_qdb4))]
dflist = [concat(a, b) for a, b, in zip(df1, df2)]

# dflist = [pd.read_pickle(at) for at in sorted(glob(glob_qdb4))]

print('loaded {} dataframes'.format(len(dflist)))

# ! crucial to load the associated datalist
with open(join(pili.root, q3, 'aux', 'crawling_datalist.pkl'), 'rb') as f:
	datalist = pickle.load(f)

dflist[0]['names']


# %%
# dflist[0]['names'].index['l_eml_fixed']
def find_index(df, name):
	return df.index[df['names']==name][-1]
model_index = find_index(dflist[0], 'l_eml')
l_eml_data = [df.iloc[model_index] for df in dflist]
m0, scale, m1, lam1, lam2 = map(np.array, zip(*[row['parameters'] for row in l_eml_data]))
w1, w2 = map(np.array, zip(*[row['weights'] for row in l_eml_data]))
par1 = {
	'lam1' : lam1,
	'lam2' : lam2,
	'w1' : w1,
	'm1' : m1
}


model_index = find_index(dflist[0], 'gmm2')
gmm2_data = [df.iloc[model_index] for df in dflist]

'median l2', np.median(lam2)
np.sqrt(2*(1/np.median(lam2))**2), 0.009

# np.corrcoef(lam1, vel)[0][1]

# %%

model_index = find_index(dflist[0], 'g_emg')
g_emg_data = [df.iloc[model_index] for df in dflist]
m0, sigma, m1, sigma, lam= map(np.array, zip(*[row['parameters'] for row in g_emg_data]))
w1, w2 = map(np.array, zip(*[row['weights'] for row in g_emg_data]))
#
par2 = {
	'sigma' : sigma,
	'lam' : lam,
	'w1' : w1
}



# %%

l_ltn_data = [df.iloc[find_index(dflist[0], 'L_LTN')] for df in dflist]
w_ltn = np.array([en['weights'][0] for en in l_ltn_data])

n_ntn_data = [df.iloc[find_index(dflist[0], 'N_NTN')] for df in dflist]
w_ntn = np.array([en['weights'][0] for en in n_ntn_data])


# plot the weights

with mpl.rc_context(usestyle):
	_shstyle = dict(element="step", fill=False, alpha=0.7, lw=4)

	fig, ax = plt.subplots(figsize=use_size)
	_xlim = (0.2, 1.0)
	nbins = 30
	sns.histplot(par2['w1'], bins=nbins, binrange=_xlim, ax=ax, color=defcolor[1], **_shstyle)
	sns.histplot(par1['w1'], bins=nbins, binrange=_xlim, ax=ax, color=defcolor[0], **_shstyle)
	# sns.histplot(w_ltn, bins=nbins, binrange=_xlim, ax=ax, **_shstyle)
	# sns.histplot(w_ntn, bins=nbins, binrange=_xlim, ax=ax, **_shstyle)
	labels = []
	labels.append(r'$g_{\sigma,\lambda}$')
	labels.append(r'$g_{\lambda,\lambda}$')
	ax.legend(labels)
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	_x, _y = ax.yaxis.get_label().get_position()
	ax.yaxis.set_label_coords(_x-0.17, _y)
	ax.set_xlabel("$w_0$", fontsize=usestyle["font.size"]+4)
	# ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	ax.xaxis.set_ticks([0.2,0.6,1.0])
	ax.tick_params(axis='x', direction='out')

if publish:
	pub.save_figure("w0_distributions")

# and check the quantiles
w1 = par1['w1']
'95% quantiles', np.quantile(w1, 0.025), np.quantile(w1, 0.5), np.quantile(w1, 0.975)

# %%
# sns.histplot(par2['sigma'], binrange=(0,0.03), **shstyle)

with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	xr = (0,200)
	sns.histplot(par1['lam1'], bins=50, binrange=xr, ax=ax, **shstyle)
	sns.histplot(par2['lam'], bins=50, binrange=xr, ax=ax, **shstyle)
	ax.legend(['l_eml', 'g_emg'])
	ax.set_xlabel("$\lambda$")



# %%
g_emg_constraint_data = [df.iloc[find_index(dflist[0], 'g_emg')] for df in dflist]
# l_nl_data = [df.iloc[find_index(dflist[0], 'L_NL')] for df in dflist]

# n_par = np.stack([en['parameters'] for en in l_nl_data])
# p_label = ["m0", "scale", "m1", "sigma", "lam2"]
# l_nl_par = {key : p_data for key, p_data in zip(p_label, n_par)}

# %%

# todo compare l_eml_fixed and g_emg_fixed for their bic
l_eml_bic = np.array([row['bic'] for row in l_eml_data])
# l_nl_bic = np.array([row['bic'] for row in l_nl_data])
g_emg_bic = np.array([row['bic'] for row in g_emg_data])
g_emg_constraint_bic = np.array([row['bic'] for row in g_emg_constraint_data])

# truncated distributions
l_ltn_data = [df.iloc[find_index(dflist[0], 'L_LTN')] for df in dflist]
l_ltn_bic = np.array([row['bic'] for row in l_ltn_data])

n_ntn_data = [df.iloc[find_index(dflist[0], 'N_NTN')] for df in dflist]
n_ntn_bic = np.array([row['bic'] for row in n_ntn_data])


# g_emg_constraint = np.array([row['bic'] for row in g_emg_data])
# gmm2_bic = np.array([row['bic'] for row in gmm2_data])

# compare gmm2 and g_emg_fixed on the whole dataset

lbest = l_eml_bic < g_emg_bic
print('compare l_eml, g_emg', lbest.sum(), lbest.size, lbest.sum()/lbest.size)

# lbest = gmm2_bic < g_emg_constraint_bic
# print('compare gmm2, g_emg_constraint', lbest.sum(), lbest.size)

# lbest = l_eml_bic < l_nl_bic
# print('compare l_eml, l_nl_bic', lbest.sum(), lbest.size)

lbest = l_ltn_bic < l_eml_bic
print('compare l_ltn, l_eml', lbest.sum(), lbest.size)
# truncated normal is still preferred by the log-likelihood

lbest = l_ltn_bic < n_ntn_bic
print('compare l_ltn, n_ntn', lbest.sum(), lbest.size)


# %%
# * analyse
# l_eml_constraint_data, l_nl_data

l_par = np.stack([en['parameters'] for en in l_eml_data])
p_label = ["m0", "scale", "m1", "lam1", "lam2"]
l_eml_par = {key : p_data for key, p_data in zip(p_label, l_par)}


# with mpl.rc_context(mplstyle):
# 	fig, ax = plt.subplots(figsize=(6,4))
# 	sns.histplot(l_eml_par['m1'], ax=ax, **shstyle)
# 	sns.histplot(l_nl_par['m1'], ax=ax, **shstyle)
# 	ax.legend(['l_eml', 'l_nl'])
# 	ax.set_xlim(-0.03, 0.03)

class Mixture(object):

	def __init__(self, mix, weights):
		self.mix = mix
		self.weights = weights

	def pdf(self, x):
		res = np.zeros_like(x)
		for w, mix in zip(self.weights, self.mix):
			res += w * mix.pdf(x)
		return res


# index = -20
index = vel_order[-2]

print('index', index, 'vel', vel[index])


# ! compare leml, lltn

em = l_eml_data[index]
m0, scale, m1, lam1, lam2 = em["parameters"]
mix = [emanalyse.Laplace(m0, scale), emanalyse.EML(m1, lam1, lam2)]
construct = Mixture(mix, em['weights'])
print('leml', 'weights', em['weights'], 'm1', m1, 'scale', scale)
data = datalist[index]

print('ks', ks_test(data, construct))

em = l_ltn_data[index]
m0, scale, m1, sigma, lam2 = em["parameters"]
mix = [emanalyse.Laplace(m0, scale), emanalyse.NL(m1, sigma, lam2)]
nl_construct = Mixture(mix, em['weights'])

print('lltn', 'weights', em['weights'], 'm1', m1, 'scale', scale)
# print(1/scale, lam2, lam1)

with mpl.rc_context({"font.size": 22}):
	fig, axes = plt.subplots(1,2,figsize = (8,4))
	ax = axes[0]
	_xlim = (-0.05, 0.10)
	handles = pub_context_plot(ax, datalist[index], construct, xlim=_xlim, full=True)
	ax.legend(handles, ["data", "Laplace", "EML", "Mixture"], fontsize=16)

	ax = axes[1]
	_xlim = (-0.05, 0.10)
	handles = pub_context_plot(ax, datalist[index], nl_construct, xlim=_xlim, full=True)
	ax.legend(handles, ["data", "Laplace", "N-L", "Mixture"], fontsize=16)
plt.tight_layout()

# %%

testing = False

if testing:
	track_index = 2924
	fig, ax = plt.subplots(figsize=(4,4))
	idx = idx_list.tolist().index(track_index)
	sns.histplot(datalist[idx], binrange=xlim, ax=ax,**shstyle)

# %%
# lets try push the l+LTN fitting away from generating a pair of laplace
# in other words prevent sigma -> 0


if testing:
	udata = udatalist[650]
	udata = udata[np.isfinite(udata)]

	print('limits', np.nanmin(udata), np.nanmax(udata))

	data = data[data!=0]
	keep = np.logical_and(data > -0.05, data < 0.10)
	data = data[keep]
	print('processed limits', data.min(), data.max())
	print('processed ', data.size, udata.size, data.size/udata.size)

	# def L_LTN(par=[0.00, 0.009, 1/0.013]):
	# construct = L_LTN([0.0, 0.09, 1/0.013])
	construct = N_NTN([0.0, 0.009, 0.009])

	with support.Timer():
		construct.fit(data, tol=1e-5, maxiter=100)
		print('iterations', construct.n_iteration)

	print('bic', construct.bic())


	print('par', construct.list_parameters())
	print('weights', construct.weights)

	with mpl.rc_context({"font.size": 22}):
		fig, ax = plt.subplots(figsize = (4,4))
		_xlim = (-0.05, 0.10)
		handles = pub_context_plot(ax, data, construct, xlim=_xlim, full=True)

# %%

truncated_normal_idx, exp_idx = idx_list[lbest], idx_list[~lbest]

index = -10
track_index = truncated_normal_idx[index]

track_index = 2924
track_index = 3069
# 760
track_index = idx_list[650]

# track_index = exp_idx[index]

print('track index', track_index)
# helpful: track_idx 3080 (-10)

idx = idx_list.tolist().index(track_index)
print('idx', idx)

em = l_ltn_data[idx]
m0, scale, m1, sigma, lam = em["parameters"]
mix = [emanalyse.Laplace(m0, scale), emanalyse.LTN(m1, sigma, lam)]
ltn_construct = Mixture(mix, em['weights'])
a, b = (0 - m1) / scale, (1e10 - m1) / scale
truncnorm = scipy.stats.truncnorm(a, b, m1, sigma)
tn_w1 = em['weights'][1]
print("mu", m1)

em = l_eml_data[idx]
m0, scale, m1, l1, l2= em["parameters"]
mix = [emanalyse.Laplace(m0, scale), emanalyse.EML(m1, l1, l2)]
lnl_construct = Mixture(mix, em['weights'])
exp = scipy.stats.expon(m1, 1/l1)
exp_w1 = em['weights'][1]
print("mu", m1)

print('bayes criterion', l_ltn_bic[idx], l_eml_bic[idx])
print("ks tests")
print('LTN', ks_test(datalist[idx], ltn_construct))
print('EML', ks_test(datalist[idx], lnl_construct))

with mpl.rc_context({"font.size": 22}):
	fig, axes = plt.subplots(1,2,figsize = (8,4))
	ax = axes[0]
	_xlim = (-0.05, 0.10)
	handles = pub_context_plot(ax, datalist[idx], ltn_construct, xlim=_xlim, full=True)
	ax.legend(handles, ["data", "Laplace", "LTN", "Mixture"], fontsize=16)

	ax = axes[1]
	handles = pub_context_plot(ax, datalist[idx], lnl_construct, xlim=_xlim, full=True)
	ax.legend(handles, ["data", "Laplace", "EML", "Mixture"], fontsize=16)

xspace = np.linspace(-0.05, 0.10, 1000)

# plot the true distributions 
with mpl.rc_context({"font.size": 22}):
	fig, ax = plt.subplots(figsize = (4,4))
	ax.plot(xspace, tn_w1 * truncnorm.pdf(xspace))
	ax.plot(xspace, exp_w1 * exp.pdf(xspace))
	ax.legend(['TN'])

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

# TODO are the median values of these two distributions similar?

# %%
# TODO !IMPORTANT compare population parameter distributions for consistency
# # so what is the "typcial" value
# m0, scale, m1, sigma, lam = np.stack([df['parameters'] for df in l_ltn_data]).T
# w0, w1 = np.stack([df['weights'] for df in l_ltn_data]).T

# print( (sigma<0.001).sum() )

# fig, ax = plt.subplots(figsize = (4,4))
# sns.histplot(lam, ax=ax, **shstyle)

# # %%
# # fig, ax = plt.subplots(figsize = (4,4))
# # sns.histplot(w0, ax=ax, **shstyle)
# # the L_LTN distribution often chooses sigma->0 

# m0, scale, m1, l1, l2 = np.stack([df['parameters'] for df in l_eml_data]).T
# w0, w1 = np.stack([df['weights'] for df in l_eml_data]).T

# fig, ax = plt.subplots(figsize = (4,4))
# # sns.histplot(m1, ax=ax, **shstyle)
# sns.histplot(w0, ax=ax, **shstyle)


# %%
# plot w1, lambda agains mean velocity
# k = 2 k means clustering

init_mean = np.array([[0.012640028, 0.46760174],[0.06321826, 0.50942003]])
km = sklearn.mixture.GaussianMixture(n_components=2, means_init=init_mean, covariance_type='full', **setting)
# km = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full', **setting)
# km = sklearn.mixture.GaussianMixture(n_components=3, covariance_type='full', **setting)
X = np.stack([vel, par1['w1']], axis=1)
labels_ = km.fit_predict(X).astype(bool)
labels_ = ~labels_
print('cluster counts', labels_.sum(), (~labels_).sum())


print(km.means_)
print(km.covariances_)
print(km.weights_)
print()

# %%

#! use the mu parameter to decide which high velocity points to redraw
mu = par1['m1']
mu_non_zero = (mu > 1e-4)

high_mean_vel = (vel > 0.06)
lltn_w0, _ = np.stack([df['weights'] for df in l_ltn_data]).T

m_scale = 1/np.median(par1['lam2'])
print('median scale', m_scale)

_, scale, _, _, _ = np.stack([df['parameters'] for df in l_ltn_data]).T
print('median scale', np.median(scale))

# %%
# * POPULATION DISTRIBUTIONS

w0, w1 = np.stack([df['weights'] for df in l_eml_data]).T
# w0, w1 = np.stack([df['weights'] for df in l_ltn_data]).T
w = w0


_xlim = (0, 0.12)

usestyle["text.latex.preamble"] = "\n".join([r"\usepackage{siunitx}"])

with mpl.rc_context(usestyle):

	mpl.rcParams.update({
		"text.usetex" : "True",
		"text.latex.preamble" : "\n".join([
			r"\usepackage{siunitx}"
		])
	})

	

	defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
	color = iter([defcolor[0], defcolor[2]])

	# scatterstyle = dict(alpha=0.2, s=10, facecolor='none', linewidth=2)
	scatterstyle = dict(alpha=0.7, s=12, facecolor='none', linewidth=1)

	fig, ax = plt.subplots(figsize=use_size)

	ax.scatter(vel, w, color=next(color), **scatterstyle)
	# ax.scatter(vel[high_mean_vel], lltn_w0[high_mean_vel], color=next(color), **scatterstyle)
	# ax.scatter(vel[labels_], w[labels_], color=next(color), **scatterstyle)
	# ax.scatter(vel[~labels_], w[~labels_], color=next(color), **scatterstyle)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	ax.set_xlim(_xlim)
	ax.set_ylim(0.2,1)
	# ax.set_ylabel("$w_0$", fontsize=32)
	ax.set_ylabel("$w_0$", fontsize=usestyle["font.size"]+4, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.26, .4)
	# ax.set_xlabel(r"mean velocity $(\unit{\mu m s^{-1}})$")
	ax.set_xlabel(r"mean velocity (\textmu ms\textsuperscript{-1})")

	idx = np.logical_and(~labels_, vel<0.08)
	m, c = np.polyfit(vel[idx], w[idx], 1)
	xspace = np.linspace(0.004, 0.10, 1000)
	fit = True
	if fit:
		ax.plot(xspace, m*xspace + c, color='k', linestyle='--', alpha=0.5, lw=2)
	
if publish:
	pub.save_figure("crawling_w0_parameter")

w1 = par1['w1']
'95% quantiles', np.quantile(w1, 0.025), np.quantile(w1, 0.5), np.quantile(w1, 0.975)

# %%
# does the w0 parameter account for all the velocity change?

a1, b1 = xspace[100], m*xspace[100] + c
a2, b2 = xspace[500], m*xspace[500] + c

(1-a1)/b1, (1-a2)/b2


# %%
# COPY FOR g_emg

_xlim = (0, 0.12)
with mpl.rc_context(mplstyle):
	defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
	color = iter([defcolor[0], defcolor[2]])
	scatterstyle = dict(alpha=0.2)
	fig, ax = plt.subplots(figsize=(4,4))

	w = par2['w1']
	ax.scatter(vel, w, color=next(color), **scatterstyle)
	# ax.scatter(vel[labels_], w[labels_], color=next(color), **scatterstyle)
	# ax.scatter(vel[~labels_], w[~labels_], color=next(color), **scatterstyle)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	ax.set_xlim(_xlim)
	ax.set_ylim(0.2,1)
	# ax.set_ylabel("$w_0$", fontsize=32)
	ax.set_ylabel("$w_0$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.26, .4)
	ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)
	# ax.set_xlabel(r"mean velocity (\textmu ms\textsuperscript{-1})")

	idx = np.logical_and(~labels_, vel<0.08)
	m, c = np.polyfit(vel[idx], w[idx], 1)
	xspace = np.linspace(0.004, 0.11, 1000)
	ax.plot(xspace, m*xspace + c, color='k', linestyle='--', alpha=0.5, lw=3)
	
# if publish:
# 	pub.save_figure("crawling_w0_parameter")


# %%
# plot the weights
with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	_xlim = (0.2, 1.0)
	w = par1['w1']
	sns.histplot(w[labels_], bins=50, binrange=_xlim, ax=ax, **shstyle)
	sns.histplot(w[~labels_], bins=50, binrange=_xlim, ax=ax, **shstyle)
	# sns.histplot(par2['w1'], bins=50, binrange=_xlim, ax=ax, **shstyle)
	ax.legend(['c1', 'c2'], fontsize=18)
	ax.set_xlabel("w1")




# %%

_xlim = (0, 0.12)
with mpl.rc_context(mplstyle):
	scatterstyle = dict(alpha=0.2)
	fig, ax = plt.subplots(figsize=(4,4))
	# x = vel[~labels_] 
	# y = par1['lam1'][~labels_]

	x = vel
	y = par1['lam1']

	ax.scatter(x, y, color=defcolor[0], **scatterstyle)

	ax.set_ylim(0, 240)
	ax.set_xlim(_xlim)

	ax.set_ylabel("$\lambda_1$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.28, .4)
	ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	# idx = y<160
	# m, c = np.polyfit(x[idx], y[idx], 1)
	# xspace = np.linspace(0.004, 0.11, 1000)
	# ax.plot(xspace, m*xspace + c, color='k', linestyle='--', alpha=0.5, lw=3)
	

if publish:
	pub.save_figure("crawling_lam2_parameter")


	
# %%

with mpl.rc_context(usestyle):
	# scatterstyle = dict(alpha=0.2, s=6)
	scatterstyle = dict(alpha=0.3, s=12, facecolor='none', linewidth=1)
	fig, ax = plt.subplots(figsize=use_size)
	lam = par1['lam2']
	ax.scatter(vel, lam, color=defcolor[0], **scatterstyle)
	# ax.scatter(vel[labels_], lam[labels_], color=defcolor[0], **scatterstyle)
	# ax.scatter(vel[~labels_], lam[~labels_], color=defcolor[2], **scatterstyle)

	ax.scatter(vel, par1['lam1'], color=defcolor[2], **scatterstyle)

	ax.set_xlim(_xlim)
	ax.set_ylim(0, 240)
	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	ax.set_ylabel("$\lambda$", fontsize=usestyle['font.size']+4, rotation=np.pi/2)
	# ax.set_ylabel("$\lambda_2$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.28, .4)
	# ax.set_xlabel("mean velocity $(\mu m/s)$")
	ax.set_xlabel(r"mean velocity (\textmu ms\textsuperscript{-1})")

	from matplotlib.lines import Line2D
	h1 = Line2D([0], [0], marker='o', color='w', markeredgecolor=defcolor[0], markersize=5, markerfacecolor='none'),
	h2 = Line2D([0], [0], marker='o', color='w', markeredgecolor=defcolor[2], markersize=5, markerfacecolor='none')
	ax.legend([h1, h2], [r'$\lambda_0$', r'$\lambda_1$'])

coef, pvalue = scipy.stats.pearsonr(vel, lam)
print(coef, pvalue)

if publish:
	pub.save_figure("crawling_lam1_parameter")

np.median(par1['lam1']), np.median(lam)

# %%

with mpl.rc_context(mplstyle):
	scatterstyle = dict(alpha=0.2)
	fig, ax = plt.subplots(figsize=(4,4))
	m1 = par1['m1']
	ax.scatter(vel, m1, color=defcolor[0], **scatterstyle)

	# ax.set_xlim(_xlim)
	# ax.set_ylim(0, 240)
	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	ax.set_ylabel("$\mu_1$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.28, .4)
	ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)

coef, pvalue = scipy.stats.pearsonr(vel, lam)
print(coef, pvalue)

if publish:
	pub.save_figure("crawling_mu1_parameter")

np.quantile(m1, 0.97)

# %%
# compare the total displacement predicted by the model and the wavelet estimate

track = track_list[index]
duration_data  = np.array([len(track) for track in track_list])

w0 = par1["w1"]
m1 = par1["m1"]
lam1 = par1['lam1']

displacement_data = m1 + 1/lam1
length_data = np.array([wave.get_distance().sum() for wave in wavelist])

# hide single track implementation
def predict_displacement(index=-12):
	m1[index] + mean_displacement 
	mean_displacement = 1/lam1
	w = w0[index]
	wave = wavelist[index]
	return wave.get_distance().sum(), (1-w) * duration * mean_displacement

predict_length = (1-w0) * duration_data * displacement_data

# score the predictions using a relative distance?

with mpl.rc_context(usestyle):
	fig, ax = plt.subplots(figsize=use_size)
	ax.axhline(1, lw=2, linestyle='--', c='k', alpha=0.40)
	predict_score = predict_length/length_data
	# scatterstyle = dict(alpha=0.2, s=6)
	scatterstyle = dict(alpha=0.7, s=12, edgecolor=defcolor[0], facecolor='none', linewidth=1)
	ax.scatter(vel, predict_score, **scatterstyle)
	ax.set_xlim(0,0.12)
	ax.set_ylim(0,2.5)
	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(3))
	ax.set_ylabel(r'$\hat{L}/L$')
	# ax.set_xlabel("mean velocity $(\mu m/s)$")
	ax.set_xlabel(r"mean velocity (\textmu ms\textsuperscript{-1})")

if publish:
	pub.save_figure("L_ratio")

print('median', np.median( predict_score))
print('mean absolute deviation', np.mean(np.abs(1-predict_score)))

# %%
# what is the ks_statistic for the l_eml_dataset

def reconstruct_nntn(em):
	em = n_ntn_data[index]
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

g_emg_construct = [reconstruct_leml(em) for em in g_emg_data]
l_eml_construct = [reconstruct_leml(em) for em in l_eml_data]
l_ltn_construct = [reconstruct_lltn(em) for em in l_ltn_data]


gemg_ks_statistic =  np.array([ks_test(datalist[i], g_emg_construct[i])[0] for i in range(len(datalist))])

leml_ks_statistic =  np.array([ks_test(datalist[i], l_eml_construct[i])[0] for i in range(len(datalist))])
lltn_ks_statistic =  np.array([ks_test(datalist[i], l_ltn_construct[i])[0] for i in range(len(datalist))])

# %%

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(vel, leml_ks_statistic, **scatterstyle)
ax.set_ylabel('k-s stat')
ax.set_xlabel('mean_velocity')

(gemg_ks_statistic < leml_ks_statistic).sum()
np.median(gemg_ks_statistic), np.median(leml_ks_statistic), np.median(lltn_ks_statistic)

# %%



# %%
# ! supposing we do exactly the same for L_LTN

m0, scale, m1, sigma, lam = np.stack([df['parameters'] for df in l_ltn_data]).T

_xlim = (0, 0.12)
with mpl.rc_context(mplstyle):
	defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
	color = iter([defcolor[0], defcolor[2]])
	scatterstyle = dict(alpha=0.2)
	fig, ax = plt.subplots(figsize=(4,4))

	ax.scatter(vel, w_ltn, color=next(color), **scatterstyle)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	ax.set_xlim(_xlim)
	ax.set_ylim(0.2,1)
	# ax.set_ylabel("$w_0$", fontsize=32)
	ax.set_ylabel("$w_0$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.26, .4)
	ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)

	# idx = np.logical_and(~labels_, vel<0.08)
	# m, c = np.polyfit(vel[idx], w_ltn[idx], 1)
	# xspace = np.linspace(0.004, 0.11, 1000)
	# ax.plot(xspace, m*xspace + c, color='k', linestyle='--', alpha=0.5, lw=3)

# %%
# sigma

_xlim = (0, 0.12)
with mpl.rc_context(mplstyle):
	scatterstyle = dict(alpha=0.2)
	fig, ax = plt.subplots(figsize=(4,4))

	x = vel
	y = sigma

	ax.scatter(x, y, color=defcolor[0], **scatterstyle)

	# ax.set_ylim(0, 240)
	ax.set_xlim(_xlim)

	ax.set_ylabel("$\sigma$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.28, .4)
	ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	# idx = y<160
	# m, c = np.polyfit(x[idx], y[idx], 1)
	# xspace = np.linspace(0.004, 0.11, 1000)
	# ax.plot(xspace, m*xspace + c, color='k', linestyle='--', alpha=0.5, lw=3)
	

# %%
# m1

_xlim = (0, 0.12)
with mpl.rc_context(mplstyle):
	scatterstyle = dict(alpha=0.2)
	fig, ax = plt.subplots(figsize=(4,4))

	vstyle = dict(alpha=0.3,c='black',linestyle='--')
	ax.axhline(0.03, lw=4, **vstyle)

	ax.scatter(vel, m1, color=defcolor[0], **scatterstyle)

	# ax.set_ylim(0, 240)
	ax.set_xlim(_xlim)

	ax.set_ylabel("$\mu_1$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.28, .4)
	ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

# %%
# check the numbers

(m1 < 0.01).sum(), (m1 > 0.01).sum()

m1[m1 > 0.01].mean(), 1.96 * scipy.stats.sem(m1[m1 > 0.01])



# %%
_xlim = (0, 0.12)
with mpl.rc_context(mplstyle):
	scatterstyle = dict(alpha=0.2)
	fig, ax = plt.subplots(figsize=(4,4))

	ax.scatter(vel, lam, color=defcolor[0], **scatterstyle)

	ax.set_ylim(0, 240)
	ax.set_xlim(_xlim)

	ax.set_ylabel("$\sigma$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.28, .4)
	ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))


# %%
# anything special about the second cluster?
c2data = [l_eml_data[i] for i, l in enumerate(labels_) if l == 1]
lam1 = par1['lam1']
fig, ax = plt.subplots(figsize=(4,4))
ax.violinplot([lam1[labels_], lam1[~labels_]])
ax.set_ylabel(r'$\lambda$', fontsize=32)


# %%
# run the remove pairs function and see how many data are removed
delta = 0.005
exp = np.array([emanalyse.remove_pairs(udata, delta=delta, explain=True) for udata in udatalist])
size = np.array([udata.size for udata in udatalist])

rmcount, fcount, rcount, plus, minus = exp.T
# with mpl.rc_context({'font.size': 20}):
# 	fig, ax = plt.subplots(figsize=(6,4))
	# sns.histplot(fcount/size, ax=ax)
	# sns.histplot(udatalist[490])
# np.median(plus/size), np.median(minus/size)
np.median(plus/size), np.median(minus/size), np.median(fcount/size)

# %%
index = 490
_rmcount, _fcount, _rcount, _plus, _minus = exp[index]
_size  = size[index]
print('+/-', _plus/_size, _minus/_size, _plus/_size * _minus/_size)
_fcount/_size
(_fcount + _rcount)/_rmcount
(_fcount + _rcount)/_size
_rmcount/_size
np.median(rmcount/size)

fraction = rmcount/size
sns.histplot(fraction)
np.median(fraction)

# %%
# examine the metadata
# what is the median spatial error estimate
'median sigma', np.median([meta['sigma'] for meta in metalist])


# %%
c1idx = np.argwhere(~labels_).ravel()
c2idx = np.argwhere(labels_).ravel()
_c2 = iter(c2idx)

# %%
# index: 8, 35, 52
# index = next(_c2)
index = 52
print('using index', index, 'track_idx', idx_list[index])

em = l_eml_data[index]
m0, scale, m1, lam1, lam2 = em['parameters']

mix = [emanalyse.Laplace(m0, scale), emanalyse.EML(m1, lam1, lam2)]
construct = Mixture(mix, em['weights'])
data = datalist[index]

cl, cr = data[data<0].size, data[data>=0].size
print('reverse/forward bias', cl/data.size, cr/data.size)

print(1/scale, lam2, lam1)


with mpl.rc_context({"font.size": 22}):
	fig, ax = plt.subplots(figsize = (4,4))
	_xlim = (-0.05, 0.10)
	handles = pub_context_plot(ax, datalist[index], construct, xlim=_xlim, full=True)
	ax.legend(handles, ["data", "Laplace", "EML", "Mixture"], fontsize=16)

if publish:
	pub.save_figure("example_blue_cluster_distribution")


# %%
forward_bias = np.array([data[data>=0].size/data.size for data in datalist])

fig, ax = plt.subplots(figsize = (4,4))

ax.violinplot([forward_bias[~labels_], forward_bias[labels_]])
ax.xaxis.set_ticks([1,2], labels=['c1', 'c2'])
ax.set_ylabel('forward bias')

# %%
idx_list.tolist().index(3069)
# %%


# * PUBLICATION
# create a publication plot which shows the raw data and donstrates the wavelet transform and local mapping
# -7, -12
# index 766 track index 3069
# index = c1idx[_idx]
index = 766
print('index', index, 'track index', idx_list[index])

track = track_list[index].copy()

short = track.cut(0,200)
# flip x,y (rotate 90)
x, y = short['x'], short['y']
_x = x - x.mean()
_y = y - y.mean()
short['x'] = -_y
short['y'] = _x

wavelet='db1'
em_config = {"wavelet":wavelet, 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}
wavemodel, lptr, meta = pwlpartition.wavelet_guess(short, config=em_config)


defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1, c2, c3 = defcolor[:3] 
c2 = '#DA5025'
c3 = '#2AA1C6'
blue = c3

model_style = {"linestyle": '-', 'lw':4, 'alpha':0.6, 'label':'wavelet', 'color':c2, "marker":'D', 'markerfacecolor' : 'none', 'markeredgewidth':2, 'markersize': 8}
ptlkw = {"linestyle":'--', 'lw':2, "marker":"o", "alpha":0.6, 'color':c3, 'markerfacecolor': 'none', 'markeredgewidth':1.5}
marker_only = model_style.copy()
marker_only["linestyle"] = 'none'


def simple_model_plot(ax, model, data):
	
	# truth_style = {"linestyle": '--', 'lw':2, 'alpha':0.5, 'label':'truth', 'color':c1}

	h1, = ax.plot(model.x, model.y, **model_style)
	h2, = ax.plot(data.x, data.y, label='data', **ptlkw)
	ax.plot(model.x, model.y, **marker_only)


	pt = np.array([0,0.05])
	ax.plot([pt[0],pt[0]+0.1], [pt[1],pt[1]], linewidth=2, c='black', alpha=0.8)
	delta = 0.005
	ax.text(pt[0]+0.1 + delta + 0.005, pt[1]-delta-0.005, r"$0.1$\textmu m", fontsize=14)

	#!  not yet defined
	_mapstyle = dict(lw=1, alpha=0.8, marker='x', markersize=6.0, markeredgewidth=2.0, color=blue, linestyle='none')
	h3, = ax.plot([], [], **_mapstyle)

	ax.axis(False)
	# ax.legend(fontsize=20, loc=(1.04, 0))
	# ax.legend([h1, h2], [r'curve, $T$', r'data, $\bm{x}$'], fontsize=12)
	ax.legend([h1, h2, h3], [r'curve, $T$', r'data, $\bm{x}$', r'map, $T(s)$'], fontsize=12)

	ax.set_aspect('equal')

# texstyle["font.family"] = "serif"
# texstyle["font.serif"] = ["MTPro2"]
# texstyle["mathtext.fontset"] = "custom"

plt.rc('text.latex', preamble=r"\usepackage{bm}")
with mpl.rc_context(texstyle):
	double_size = 1.1 * np.array([2*column_width, 2*column_width])
	fig, ax = plt.subplots(figsize=double_size)
	simple_model_plot(ax, wavemodel, lptr)

if publish:
	pub.save_figure("example_wavelet_track")	

# fig, ax = plt.subplots(figsize=(4,4))
# sns.histplot(datalist[index], ax=ax, lw=6, **shstyle)

# %%
# * COMPUTE the curve coordinates
curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr, ln=11)
curve = mdl.LPshape(wavemodel.x, wavemodel.y)

# %%
# plot the same again but this time map data 


with mpl.rc_context(texstyle):
	fig, ax = plt.subplots(figsize=double_size)
	model = wavemodel

	# model_style = {"linestyle": '-', 'lw':5, 'alpha':0.6, 'label':'wavelet', 'color':c2}
	# ptlkw = {"linestyle":'none', 'lw':2, "marker":"o", "alpha":0.4, 'color':c3}
	c3 = '#2AA1C6'
	c3 = 'gray'
	ptlkw = {"linestyle":'none', 'lw':1, "marker":"o", "alpha":0.6, 'color':c3, 'markerfacecolor': 'none', 'markeredgewidth':1.5}
	# c3 = '#2AC67B'

	# mapstyle = dict(linestyle='none', marker='o', color=c3, alpha=0.4)
	mapstyle = ptlkw.copy()
	mapstyle.update(dict(lw=2, alpha=0.8, marker='x', markersize=6.0, markeredgewidth=2.0, color=blue))
	maplstyle = dict(linestyle='-', lw=1.5, color=c3, alpha=0.5)

	h2, = ax.plot(model.x, model.y, **model_style)
	h1, = ax.plot(lptr.x, lptr.y, label='data', **ptlkw)

	ax.axis(False)
	ax.set_aspect('equal')

	# plot the mapping lines
	for i, pt in enumerate(lptr.get_n2()):
		x, y = pt
		u = curve_coord[i]
		mx, my = curve(u)
		ax.plot([x, mx], [y, my], **maplstyle)
		h3, = ax.plot(mx, my, **mapstyle)

	ax.plot(model.x, model.y, **marker_only)

	# ax.legend([h1, h2, h3], ['data', 'wavelet', 'mapped'], fontsize=20, loc=(1.04, 0))

	from matplotlib.lines import Line2D
	h3 = Line2D([0], [0], marker='o', color='w', markerfacecolor=defcolor[0], markersize=10),
	# ax.legend([h3], ['mapped'], fontsize=20, loc=(1.04, 0))

if publish:
	pub.save_figure("example_mapped_data")	



# %%

def pub_context_plot(ax, data, mm, xlim=_xlim, full=True, extra={}):
	defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
	lstyle = dict(alpha=0.8, lw=3, linestyle='--')
	partstyle = dict(alpha=0.7, lw=3)
	shstyle = dict(element="step", fill=False, alpha=0.5, lw=6)
	vstyle = dict(alpha=0.2,c='black',linestyle='--', lw=2)
	ax.axvline(0, **vstyle)

	sns.histplot(data, ax=ax, stat='density', **shstyle)
	h1 = ax.lines[-1]

	xspace = np.linspace(data.min(), data.max(), 1000)
	color = itertools.cycle([defcolor[0]])

	color = iter(defcolor[1:])
	hlist = []
	for i, mix in enumerate(mm.mix):
		h2, = ax.plot(xspace, mm.weights[i] * mix.pdf(xspace), color=next(color), **partstyle) 
		hlist.append(h2)
	if full:
		pdf = np.stack([mm.weights[i] * mix.pdf(xspace) for i, mix in enumerate(mm.mix)]).sum(axis=0)
		h3, = ax.plot(xspace, pdf, color="#1B689D", **lstyle)

	ax.set_xlim(xlim)
	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	ax.set_ylabel("Density")
	ax.set_xlabel("displacement $(\mu m)$")
	if full:
		return [h1, *hlist, h3]
	else:
		return [h1, *hlist]


# * Plot publication example distribution 
index = -12
print('index', index, 'track index', idx_list[index])
_xlim = (-0.05, 0.10)

em = l_eml_data[index]
print('bic', em['bic'])
m0, scale, m1, lam1, lam2 = em['parameters']
w1, w2 = em['weights']
mix = [emanalyse.Laplace(m0, scale), emanalyse.EML(m1, lam1, lam2)]
construct = Mixture(mix, em['weights'])


eml_label = r'$f_{|\lambda|} \ast f_{\lambda}$'

with mpl.rc_context(usestyle):
	a, b, = use_size
	local_use_size = (1.22 * a , b)
	fig, ax = plt.subplots(figsize=local_use_size)
	data = datalist[index]
	data = data[data!=0]
	handles = pub_context_plot(ax, data, construct, full=True)
	ax.legend(handles, ["data", "Laplace", eml_label, "mixture"], handlelength=1, loc=(0.43,0.38))
	ax.set_yticks([])
	ax.set_ylabel('')


if publish:
	pub.save_figure("em_fit_laplace")

# * Statistical test to see if the data and the predicted distribution match
# we didn't bother to implement cdf for all the functions
# so lets just compute it numerically
'ks_test', ks_test(datalist[index], construct)

#  so pvalue < 0.05 means that we should actually reject the null hypothesis that the distributions are the same


# %%

# em = l_nl_data[index]
# print('bic', em['bic'])
# em["parameters"]

# m0, scale, m1, sigma, lam = em['parameters']
# w1, w2 = em['weights']
# # print(1/scale, lam)

# mix = [emanalyse.Laplace(m0, scale), emanalyse.NL(m1, sigma, lam)]
# construct = Mixture(mix, em['weights'])


# with mpl.rc_context({"font.size": 22}):
# 	fig, ax = plt.subplots(figsize = (4,4))
# 	data = datalist[index]
# 	data = data[data!=0]
# 	handles = pub_context_plot(ax, data, construct, full=True)
# 	ax.legend(handles, ["data", "Laplace", "N-L", "Mixture"], fontsize=16)

# if publish:
# 	pub.save_figure("em_fit_l_nl")

# 'ks_test', ks_test(datalist[index], construct)
# # why ks_statistic so high here?


# %%
# lets plot g_emg_fixed for the same index

em = g_emg_data[index]
print('bic', em['bic'])
m0, sigma, m1, sigma, lam = em['parameters']
w1, w2 = em['weights']

mix = [emanalyse.Gauss(m0, sigma), emanalyse.EMG(m1, sigma, lam)]
construct = Mixture(mix, em['weights'])

emg_label = r'$f_{\sigma} \ast f_{\lambda}$'

with mpl.rc_context(usestyle):

	fig, ax = plt.subplots(figsize = local_use_size)
	data = datalist[index]
	data = data[data!=0]
	handles = pub_context_plot(ax, data, construct)
	ax.legend(handles, ["data", "Normal", emg_label, "mixture"], handlelength=1, loc=(0.43,0.38))


	ax.set_yticks([])
	ax.set_ylabel('')

if publish:
	pub.save_figure("em_fit_normal")

'ks_test', ks_test(datalist[index], construct)

# %%

gmm = sklearn.mixture.GaussianMixture(n_components=2)
gm = gmm.fit(datalist[index].reshape(-1,1))
print(gm.weights_)
print(gm.means_)
print(gm.covariances_)

# %%
# run gmm models
gmmlist = []

with support.Timer():
	for data in datalist:
		gmm = sklearn.mixture.GaussianMixture(n_components=2)
		gm = gmm.fit(data.reshape(-1,1))
		gmmlist.append(gm)

# %%

gmm_means = np.array([gm.means_ for gm in gmmlist])
gmm_weights = np.array([gm.weights_ for gm in gmmlist])
gmm_covar = np.array([gm.covariances_ for gm in gmmlist])

midx = np.array([np.abs(m).argmin() for m in gmm_means])
m_min = np.array([m.min() for m in gmm_means])
m_var = np.array([v[midx[i]][0,0] for i, v in enumerate(gmm_covar)])
print('mu', m_min.min(), np.median(m_min), m_min.max())
print('mu 95%', np.quantile(m_min, 0.025), np.median(m_min), np.quantile(m_min, 0.975))
m_std = np.sqrt(m_var)
print('sigma', m_std.min(), np.median(m_std), m_std.max())

print("mean absolute deviation", np.mean(np.abs(m_min)))

# %%
gmm_means[-12]

# %%
# lets plot gmm2 for the same index

# pub.set_write_dir(join(pili.root, '../sparsel/EM_paper/'))
# pub.set_write_dir(join(pili.root, '../thesis/'))

em = gmm2_data[index]
print('bic', em['bic'])
m0, sigma0, m1, sigma1 = em['parameters']
w1, w2 = em['weights']
print('weights', w1, w2)
print('mu, sigma', m0, sigma0)
print('mu, sigma', m1, sigma1)

mix = [emanalyse.Gauss(m0, sigma0), emanalyse.Gauss(m1, sigma1)]
construct = Mixture(mix, em['weights'])

# with mpl.rc_context({"text.usetex": False, "font.size": 22}):

local_use_size = (4,4)
with mpl.rc_context(thesis.texstyle):
	_xlim = (-0.06, 0.1)
	fig, ax = plt.subplots(figsize = local_use_size)
	data = datalist[index]
	data = data[data!=0]
	handles = pub_context_plot(ax, data, construct, xlim=_xlim)
	labels = [r"data", r"$\mu = 0$", r"$\mu = 0.006$", r"mixture"]
	ax.legend(handles, labels, 
		loc=(0.53,0.43), framealpha=1.0, handlelength=1, frameon=True, fontsize=20)
	ax.set_xlabel(r"displacement (\textmu m)")

if publish:
	pub.save_figure("em_fit_gmm2")
	thesis.save_figure("em_fit_gmm2")

'ks_test', ks_test(datalist[index], construct)


# %%
# !lets plot the equivalent true distribution

# first get the true sigma
m_sigma = np.sqrt(sigma1**2 - sigma0**2)
sigma0, sigma1, m_sigma 

# %%

x1 = np.linspace(*xlim, num=2000)
pdf2 = scipy.stats.norm(m1, m_sigma).pdf(x1)
with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	# ax.axvline(0, **vstyle)
	ax.set_xlim(xlim)

	# width = 0.0062
	# height = w1/width
	# rect = mpl.patches.Rectangle((-width/2,width/2), width, height, alpha=0.8, fill=None, lw=4, color=defcolor[1])
	# h1 = ax.add_patch(rect)
	h1, = ax.plot([0,0], [0,60], lw=8,  color=defcolor[1])
	h2, = ax.plot(x1, w2*pdf2, lw=4, color=defcolor[2])
	
	# ax.xaxis.set_major_locator(plt.MaxNLocator(3))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	ax.xaxis.set_ticks([-0.04, 0, 0.04, 0.08])
	ax.set_ylim(0,62)

	ax.legend([h1, h2], [r'$\delta_{x=0}$', r'$\mathcal{N}(\mu, \sigma)$'], fontsize=18)
	ax.set_xlabel(r'displacement (\textmu m)')
	ax.set_ylabel('Density')

	# !composite
	ax.set_ylabel('')
	ax.yaxis.set_ticks([])

pub.save_figure('gmm2_true_estimate')

# plt.tight_layout()
present_dir = join(pili.root, '../sparseml/EM_paper/impress/ims')
if publish:
	plt.savefig(join(present_dir, 'gmm2_true_estimate.png'))
	thesis.save_figure("gmm2_true_estimate")


# %%
#! another presentation plot
with mpl.rc_context(mplstyle):

	fig, ax = plt.subplots(figsize=(4,4))
	ax.set_xlim(xlim)
	ax.xaxis.set_ticks([-0.04, 0, 0.04, 0.08])

	ax.axvline(0, lw=2, c='k', alpha=0.6, linestyle='--')


	entries, bins = np.histogram(data, bins='auto')
	bins = bins[:-1] + np.diff(bins)/2
	lwstyle = dict(alpha=0.6, lw=8)
	ax.plot(bins, entries, lw=2)

	lidx = bins<0
	fill_style = dict(hatch='\\', facecolor='r', alpha=0.1)
	ax.fill_between(bins[lidx], np.zeros(int(lidx.sum())), entries[lidx]+1, **fill_style)

	ax.set_xlabel('displacement $(\mu m)$')
	ax.set_ylabel('Count')

	ax.set_ylim((0,None))

if publish:
	plt.tight_layout()
	plt.savefig(join(present_dir, 'shaded_left_side.png'))




# %%
# TODO
# plot population distributions of two component GMM parameters 

m0, sigma0, m1, sigma1 = map(np.array, zip(*[row['parameters'] for row in gmm2_data]))
w1, w2 = map(np.array, zip(*[row['weights'] for row in gmm2_data]))

print(np.median(sigma0), np.median(m1), np.median(sigma1))

# sns.histplot(sigma0)
# sns.histplot(sigma1)
# sns.histplot(m1)
_xlim = (0, 0.12)
with mpl.rc_context(mplstyle):
	color = iter([defcolor[0], defcolor[2]])
	scatterstyle = dict(alpha=0.2)
	fig, ax = plt.subplots(figsize=(4,4))

	ax.scatter(vel, w1, **scatterstyle)
	ax.set_xlim(_xlim)


# %%



# %%
# * FOR TESTING PROPROCESSING
# load the preprocessing test and analyse the fit parameters

path = join(pili.root, 'notebook/em_algorithm/test_preprocessing/*.pkl')
pdlist = [pd.read_pickle(at) for at in sorted(glob(path))]

def unpack(em, var):
	return np.stack([l for l in em[var]])

qarr = np.stack([em['q'] for em in pdlist])
delta_arr = np.stack([em['delta'] for em in pdlist])
weights = np.stack([unpack(em, 'weights') for em in pdlist])
parameters = np.stack([unpack(em, 'parameters') for em in pdlist])
w0 = weights[:,:,0]
# 0, 1, 2, 3, 4
l1 = parameters[:,:,3]

fig, ax = plt.subplots(figsize=(4,4))
# ax.violinplot(list(l1.T))

scatterstyle = dict(alpha=0.2)
ax.scatter(vel, l1.T[0], color=defcolor[2], **scatterstyle)
for i in range(6):
	q = qarr[0,i]
	delta = delta_arr[0,i]
	print(q, delta, (l1.T[i] > 150).sum() )

# %%
_xlim = (0, 0.12)
with mpl.rc_context(mplstyle):
	defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
	color = iter([defcolor[0], defcolor[2]])
	scatterstyle = dict(alpha=0.2)
	fig, ax = plt.subplots(figsize=(4,4))

	w = w0.T[5]
	ax.scatter(vel, w, color=next(color), **scatterstyle)
	# ax.scatter(vel[labels_], w[labels_], color=next(color), **scatterstyle)
	# ax.scatter(vel[~labels_], w[~labels_], color=next(color), **scatterstyle)

	ax.xaxis.set_major_locator(plt.MaxNLocator(4))
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	ax.set_xlim(_xlim)
	ax.set_ylim(0.2,1)
	# ax.set_ylabel("$w_0$", fontsize=32)
	ax.set_ylabel("$w_0$", fontsize=32, rotation=np.pi/2)
	ax.yaxis.set_label_coords(-0.26, .4)
	ax.set_xlabel("mean velocity $(\mu m/s)$", fontsize=20)

	# idx = np.logical_and(~labels_, vel<0.08)
	# m, c = np.polyfit(vel[idx], w[idx], 1)
	# xspace = np.linspace(0.004, 0.11, 1000)
	# ax.plot(xspace, m*xspace + c, color='k', linestyle='--', alpha=0.5, lw=3)
	


# %%
