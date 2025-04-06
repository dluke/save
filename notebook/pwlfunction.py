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
# Evaluating 1d PWL and piecewise constant functions 

# %% 
import os
import random
import numpy as np
import pickle
import json
from glob import glob
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
pi = np.pi

from copy import copy, deepcopy

import scipy
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from skimage.restoration import denoise_wavelet

# import sctml
# import sctml.publication as pub
# print("writing figures to", pub.writedir)
import thesis.publication as thesis

import pili
from pili import support
import mdl
import synthetic
import pwltree

from pwcsearch import *
from pwcshuffle import *

images_dir = join(pili.root, "../sparseml/images")

# %% 
#! setup
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']

from functools import wraps
def history(f):
	value = {}
	@wraps(f)
	def history_function(k):
		if k in value:
			return value[k]
		else:
			result = f(k)
			value[k] = result
			return result
	history_function.value = value
	return history_function

# %% 
#! piecewise constant function has form [[t, y(t)], ...] with first element [0, y(0)]

rng = np.random.default_rng(0)

# !parameters
sigma = 0.2
param = {
	'sigma': 0.2,
	'min_separation': 0.3
	}
pwc = generate_pwc(param, rng)
space, sample_data = sampling(pwc, param, rng)

def plot_pwc(ax, t_data, y_data, style={}):
	l =  ax.plot(np.repeat(t_data,2)[1:-1], np.repeat(y_data[:-1],2), **style)

c3 = '#2AA1C6'
ptlkw = {"linestyle":'--', 'lw':2, "marker":"o", "alpha":0.6, 'color':c3, 'markerfacecolor': 'none', 'markeredgewidth':1.5}
def plot_samples(ax, space, samples, style=ptlkw):
	ax.plot(space, samples, **style)

fig, ax = plt.subplots(figsize=(6,4))
plot_pwc(ax, pwc.t_data, pwc.y_data)
plot_samples(ax, space, sample_data)


# %%
# ! use pwltree solver 

solver = pwltree.new_pwc_tree_solver(sample_data)
solver.build_max_tree()
solver.build_priority()
solver.solve(pwltree.stop_at(2 * sigma**2))

pwc_solution = get_pwc_model(solver)
index, y_data = zip(*pwc_solution)
t_data = space[np.array(index)]

fig, ax = plt.subplots(figsize=(6,4))
solve_style = {'linestyle':'--', 'lw':2.5}
plot_pwc(ax, t_data, y_data, solve_style)
plot_samples(ax, space, sample_data)

# %%
#! parameter analysis
param = {
	'sigma': 0.2,
	'min_separation': 0.3,
	'err_lambda': 2
	}

# new random state
random_state = np.random.default_rng()

#! 1. minimum separation

var_min_separation = np.linspace(0.0, 0.5, 6, True)
variant = ('min_separation', np.linspace(0.0, 0.5, 6, True))


# %%
# !replicates
N = 2
stats, pwc_functions, param_record = linear_search(variant, N, param)
count_measure = [np.sum(stat_data['abs_count'])/np.sum(stat_data['count_segments']) for stat_data in stats]

# %%
#! use async data
data_directory = join(pili.root, "../sparseml/run/pwconstant/min_separation/")


def load_pickle(file):
	with open(file, 'rb') as f:
		return pickle.load(f)

def load_json(file):
	with open(file, 'r') as f:
		return json.load(f)


def load_solve_data(at):
	return [load_pickle(file) for file in sorted(glob(join(at, 'data_*')))]
def load_pwc(at):
	return [load_pickle(file) for file in sorted(glob(join(at, 'pwc_*')))]
def load_param(at):
	return [load_json(file) for file in sorted(glob(join(at, 'param_*')))]
	
solve_data_ll = load_solve_data(data_directory)
pwc_data_list = load_pwc(data_directory)
param_list = load_param(data_directory)

# %%


def compile_stats(pwc_data_list, solve_data_ll, metrics):
	def all_metric(pwc, solve_data): 
		return [metric(pwc, solve_data) for metric in metrics]
	stats = []
	for pwc_list, solve_data_list in zip(pwc_data_list, solve_data_ll):
		m_data = [all_metric(pwc, solve_data) for pwc, solve_data in zip(pwc_list, solve_data_list)]
		_stat = label_data( list(map(np.array, list(zip(*m_data)))), metrics) 
		stats.append(_stat)
	return stats
stats = compile_stats(pwc_data_list, solve_data_ll, global_metrics)

def load_and_compile_stats(data_directory, metrics=global_metrics):
	pwc_data_list = load_pwc(data_directory)
	solve_data_ll = load_solve_data(data_directory)
	return compile_stats(pwc_data_list, solve_data_ll, metrics)

stats = load_and_compile_stats(data_directory)

# %%

def bootstrap_interval(data, statistic=np.mean):
	res = scipy.stats.bootstrap((data,), statistic, vectorized=True)
	interval = res.confidence_interval
	return statistic(data) - interval.low, interval.high -statistic(data)

count_measure = [np.sum(stat_data['abs_count'])/np.sum(stat_data['count_segments']) for stat_data in stats]
err_interval = np.array([bootstrap_interval(stat_data['abs_count']/stat_data['count_segments']) for stat_data in stats]).T

# %%
# !plot
fig, ax = plt.subplots(figsize=(4,4))
style = dict(marker='s', markersize=8, markerfacecolor='none', markeredgewidth=1.5)
ax.plot(variant[1], count_measure, **style)
ax.errorbar(variant[1], count_measure, yerr=err_interval, fmt='none')
ax.set_ylabel("error ratio")
ax.set_xlabel("min separation")
ax.set_ylim((0,None))


# %%
#! parameter 2 sigma

variant = ('sigma', np.linspace(0.1, 0.5, 5, True))
# stats, pwc_functions, param_record = linear_search(variant, N, param)

data_directory = join(pili.root, "../sparseml/run/pwconstant/sigma/")
stats = load_and_compile_stats(data_directory)

# %%
count_measure = [np.sum(stat_data['abs_count'])/np.sum(stat_data['count_segments']) for stat_data in stats]
err_interval = np.array([bootstrap_interval(stat_data['abs_count']/stat_data['count_segments']) for stat_data in stats]).T


# %%
# !plot
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(variant[1], count_measure, **style)
ax.errorbar(variant[1], count_measure, yerr=err_interval, fmt='none')

ax.set_ylabel("error ratio")
ax.set_xlabel("sigma")
ax.set_ylim((0,None))

# %%
#! pull a solution from the drive for testing 
pwc_data_list = load_pwc(data_directory)
solve_data_ll = load_solve_data(data_directory)
param_list = load_param(data_directory)

param_index = -1
pwc = pwc_data_list[param_index][0]
param = param_list[param_index]
print('param', param)

# index, t_data, y_data = solve_data_ll[param_index][0]
space, sample_data = sampling(pwc, param)
index, t_data, y_data = solve(param, space, sample_data)

fig, ax = plt.subplots(figsize=(6,4))
plot_pwc(ax, pwc.t_data, pwc.y_data)
plot_samples(ax, space, sample_data)
plot_pwc(ax, t_data, y_data, solve_style)

# %%
#! generate a solution using PWLtree for testing
param = {
	'sigma': 0.5,
	'min_separation': 1.0,
	'err_lambda' : 1
	}

m = param.get('m', 10)
min_separation = param.get('min_separation', 0)
def random_alternating(rng):
	return 1.0 if rng.random() < 0.5 else -1.0

def rs_ra(rng):
	# random size random direction
	u = rng.uniform(0.1, 1.0)
	return u if rng.random() < 0.5 else -u

def myosinVI(rng):
	# random size random direction
	u = rng.uniform(0.5, 1.0)
	return u if rng.random() < 0.7 else -u


#! fix randomness
# rng = np.random.default_rng(0)
# pwc = PWCfunction(random_pwc(m, loc=min_separation, y_sample=rs_ra, rng=rng))

rng = np.random.default_rng(2)
pwc = PWCfunction(random_pwc(m, loc=min_separation, y_sample=myosinVI, rng=rng))

# tree solver
space, sample_data = sampling(pwc, param, rng)
index, t_data, y_data = solve(param, space, sample_data)

# wavelet transform
wave_config = {"wavelet":'db1', 'method':'BayesShrink', "mode":'soft', "rescale_sigma":False}
wave = denoise_wavelet(sample_data, sigma=param.get('sigma'), **wave_config)

fig, ax = plt.subplots(figsize=(6,4))
plot_pwc(ax, pwc.t_data, pwc.y_data)
plot_samples(ax, space, sample_data)
# plot_pwc(ax, t_data, y_data, solve_style)
ax.plot(space, wave, **solve_style)



# %%

def split_samples(pwc, sample_data):
	# split sample data into blocks according to its truth function
	dt = 0.1
	sample_t = np.arange(0, pwc.t_data[-1], dt)
	partition = np.searchsorted(sample_t, pwc.t_data)
	return np.split(sample_data, partition[1:-1])

def sample_probabilities(pwc, sample_data, sigma):
	# get the probability associated with each block after random generation
	result = []
	for i, samples in enumerate(split_samples(pwc, sample_data)):
		residual = np.abs(samples - pwc.y_data[i])
		p = residual_probability(residual, sigma)
		result.append(p)
	return result

sample_p = sample_probabilities(pwc, sample_data, param.get('sigma'))
print('read out generated probabilities')
[round(p, 3) for p in sample_p]


# TODO record the history of the optimiser

# %%
print('INITIAL GUESS')
fig, ax = plt.subplots(figsize=(6,4))
plot_pwc(ax, pwc.t_data, pwc.y_data)
plot_samples(ax, space, sample_data)
plot_pwc(ax, t_data, y_data, solve_style)


# %%

from pwltree import PWCLoss, mean_squared_error, PWCMinimizeLoss

def probability_threshold(ml, param):
	sigma = param.get('sigma')
	return residual_probability(ml.loss.get_residual(), sigma)

data = (space, sample_data)
shuffle = ShuffleSolver(data, index,
	Loss = PWCLoss,
	evaluate_loss=probability_threshold,
	MinimizeLoss=PWCMinimizeLoss,
	param=param
	)

shuffle.init()
probabilities = shuffle.get_probabilities(param.get('sigma'))
probabilities


# %%
# threshold =  2 * param.get('sigma')**2
threshold = 0.85
# threshold = 0.90
print('threshold', threshold)

DEBUG = False
N = 5000
shuffle.annealing(probability_threshold, threshold , N=N) 
print('finished annealing')
N = 5000
shuffle.annealing(probability_threshold, threshold , N=N, start_t=0.0) 
print('finished')

# shuffle.random_solve(probability_threshold, threshold , N=N) 

# save copy
# _shuffle = deepcopy(shuffle)

# %%
# TODO random moves should optimise likelihood or MSE

# ml = shuffle.evaluate_join(2)
# probability_threshold(ml, param)
# shuffle.random_move(probability_threshold, threshold, N=1000)
# DEBUG = True
# shuffle.minimize_at(1)
# shuffle.join(2)
# ml = shuffle.evaluate_join(14)
# probability_threshold(ml, param)
# shuffle.random_gss()

# %%
print(shuffle.get_probabilities(param.get('sigma')))
print()

fig, ax = plt.subplots(figsize=(6,4))
plot_pwc(ax, pwc.t_data, pwc.y_data)
plot_samples(ax, space, sample_data)
p, t, y = shuffle.get_model()
print('partition', p)
print('t', t)
print('y', y)
plot_pwc(ax, t, y, 
	{**solve_style, **dict(color=defcolor[1])}
	)

# %%
prob = shuffle.get_probabilities(param.get('sigma'))
for pair in zip(p[:-1], prob):
	print(pair)	

# %%
setup = list(zip([plot_pwc, plot_samples, plot_pwc],
	[(pwc.t_data, pwc.y_data), (space, sample_data), (t, y)],
	[{}, ptlkw, {**solve_style, **dict(color=defcolor[1], lw=3)}]
))
def plot_triple(setup):
	with mpl.rc_context(thesis.basestyle):
		fig, axes = plt.subplots(1, 3, figsize=(18,4), sharey=True)
	def plot_set(ax, tup):
		plot_function, plot_data, style = tup 
		plot_function(ax, *plot_data, style=style)
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		
	ax = axes[0]
	plot_set(ax, setup[0])
	ax = axes[1]
	plot_set(ax, setup[1])
	ax = axes[2]
	plot_set(ax, setup[0])
	plot_set(ax, setup[1])
	plot_set(ax, setup[2])
	fig.tight_layout()
	

plot_triple(setup)
# support.save_figure("pwc_chi2_myosinVI", target_dir=images_dir)
# support.save_figure("pwc_chi2_varheight_0.90", target_dir=images_dir)
# support.save_figure("pwc_chi2_solve_example", target_dir=images_dir)
# support.save_figure("pwc_chi2_solve_alternating", target_dir=images_dir)



# %%
#! implement maximum likelihood solver

from pwltree import PWCLoss, mean_squared_error, PWCMinimizeLoss

data = (space, sample_data)
shuffle = ShuffleSolver(data, index,
	Loss = PWCLoss,
	evaluate_loss=mean_squared_error,
	MinimizeLoss=PWCMinimizeLoss,
	param= {} #TODO
	)

shuffle.init()
threshold =  2 * param.get('sigma')**2
print('threshold', threshold)
shuffle.random_solve(threshold , N=100) 




# %%
#! try on twitching data
import _fj
track = _fj.trackload_original([2924])[0]
from skimage.restoration import estimate_sigma
hat_sigma =  np.mean([estimate_sigma(track['x']), estimate_sigma(track['y'])])
short = track.cut(0,200)

from pwltree import Loss2D, MinimizeLoss, mean_squared_error, stop_at

data = np.column_stack([short['x'], short['y']])
print('estimate sigma', hat_sigma)

solver = pwltree.TreeSolver(data, params={}, 
	overlap=True, 
	Loss=Loss2D, 
	MinimizeLoss=MinimizeLoss,
	evaluate_loss=mean_squared_error,
	)

solver.build_max_tree()
solver.build_priority()
solver.solve(pwltree.stop_at( 2 * hat_sigma**2 ))
lines = solver.get_disconnected_model()


# %%


def plot_lines(ax, lines):
	for line in lines:
		p1, p2 = line
		x1, y1 = p1
		x2, y2 = p2
		ax.plot([x1, x2], [y1, y2], marker='D', markersize=10, markerfacecolor='none', markeredgewidth=2.5)
	ax.set_aspect("equal")

fig, ax = plt.subplots(figsize=(10,10))
plot_samples(ax, solver.point_data.T)
plot_lines(ax, lines)



# %%
#! shuffle solver

index = solver.get_partition()
data = (short['time'], solver.point_data)
shuffle = ShuffleSolver(data, index,
	Loss = Loss2D,
	evaluate_loss=mean_squared_error,
	MinimizeLoss=MinimizeLoss
	)

shuffle.init()

threshold =  4 * hat_sigma**2
print('threshold', threshold)
shuffle.random_solve(threshold , N=100) 

p, t, y = shuffle.get_model()
y
# %%
print(hat_sigma, threshold)
shuffle.init()
np.array(shuffle.loss_data[:-1]) < threshold
for i, n in enumerate(np.diff(shuffle.partition)):
	print(i, n)
list(enumerate(shuffle.get_probabilities(hat_sigma)))
shuffle.loss_data[:-1]/threshold

# %%
_shuffle = deepcopy(shuffle)

# %%
_shuffle.join(7)

# %%

lines = get_disconnected_model(_shuffle)

fig, ax = plt.subplots(figsize=(10,10))
plot_samples(ax, solver.point_data.T)
plot_lines(ax, lines)

print(list(enumerate(shuffle.partition)))
list(enumerate(_shuffle.get_probabilities(hat_sigma)))



# %%
#! why are these probabilities so small?
#! we can experiment by computing N i.i.d numbers from normal distribution
# # scipy.stats.chi2(19, scale=0.5).rvs(10)

# v = []
# for i in range(10000):
# 	N = 5
# 	randn = scipy.stats.norm(scale=0.5).rvs(N)
# 	chi_value = np.sum((randn/0.5)**2)
# 	v.append( scipy.stats.chi2.cdf(chi_value, N) )
# np.mean(v)

# %%
#! TEST likelihood loss function: generate a single regression line 

# scale = 0.1
# N = 100001

# Y = scipy.stats.norm(loc=1, scale=scale).rvs(N)
# T = np.linspace(0, 1, N)
# fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(T, Y, **ptlkw)
# ax.set_ylim(0.0, 2.0)

# Y = scipy.stats.norm(loc=1, scale=scale).rvs(N)
# loss = pwltree.PWCLoss(Y, scale)
# loss(1)


# %%

# plot the annealing function
def test_accept_function():
	t = np.linspace(0,1.0,1000, True)
	_T = 0.5
	p_k = 0.75
	f = np.array([accept_function(x, p_k=p_k, T=_T) for x in t])
	fig, ax = plt.subplots(figsize=(4,4))
	# ax.plot(t,f)
	ax.axvline(p_k, linestyle='--', c='k', alpha=0.3)

	for T in np.linspace(0,1.0,5):
		f = np.array([accept_function(x, p_k=p_k, T=T) for x in t])
		ax.plot(t, f, alpha=0.5)