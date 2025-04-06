
# a script for running EM analysis on the crawling dataset

import os
import sys
import time
import json
from tqdm import tqdm
import numpy as np
import scipy.stats
import scipy.optimize
import pickle
join = lambda *x: os.path.abspath(os.path.join(*x))
import pandas as pd
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import pili
from pili import support
import emanalyse
from emanalyse import Gauss, EMG, MixtureModel
import readtrack
import _fj
import pwlpartition


# ----------------------------------------------------------------
import argparse
parse = argparse.ArgumentParser()
parse.add_argument('-n', '--n_sample', help="resample rate", type=int, default=1)
parse.add_argument('-m', '--maxiter', help="max EM iterations", type=int, default=100)
parse.add_argument('-j', '--maxcores', help="max cores", type=int, default=8)
args = parse.parse_args()


# ----------------------------------------------------------------
# control flags

preprocess_only = False

setting_maxiter = args.maxiter
# setting_maxiter = 10

remove_zeros = True
remove_pairs = True

test = False
single = False
exit_early = False
verbose = False

# "multiscale iteration 01, sampling trajectories at 1 step
# sample_list = [1,2,3,5,10]
# n_sample = sample_list[0]
n_sample = args.n_sample
print('using n_sample', n_sample)
print('maxiter ', setting_maxiter)
print('maxcores ', args.maxcores)

data_dir = join('./', 'm01_sample_{:02d}/'.format(n_sample))
if not os.path.exists(data_dir):
	print('create directory ', data_dir)
	os.makedirs(data_dir)

print('using directory ', data_dir)

# ----------------------------------------------------------------
# preprocessing the data

# analyse the crawling dataset
# * LOAD the full dataset

wavelet='db1'
em_config = {"wavelet":wavelet, 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}

load_path = join(pili.root, 'notebook/thesis_classification/crawling.npy')
idx_list = np.loadtxt(load_path).astype(int)
if single: 
	idx_list = [idx_list[154]]
track_list = _fj.trackload_original(idx_list)

print("Loaded {:d} trajectories from {}".format(len(idx_list), load_path))

def get_data(track):
	wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
	curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr)
	udata = np.diff(curve_coord) 
	return udata, wavemodel, lptr, meta

def resample(track, n=10):
	track = track.copy()
	data = track._track.copy()
	return readtrack.TrackLike(data[::n])

def fixed_process(data, xlim=(-0.10, 0.10), delta=0.005, remove_zeros=True, remove_pairs=False):
	data = data[np.isfinite(data)]
	if remove_zeros:
		data = data[data!=0]
	if remove_pairs:
		data = emanalyse.remove_pairs(data ,delta=delta)
	keep = np.logical_and(data > xlim[0], data < xlim[1])
	return data[keep]


# q = 0.005
delta = 0.005
xlim = (-0.10, 0.10)

# datalist = [fixed_process(udata, xlim, delta, remove_zeros=remove_zeros, remove_pairs=remove_pairs) for udata in udatalist]

# write these data
auxdir = join(data_dir, 'aux/')
if not os.path.exists(auxdir):
	print('create directory ', auxdir)
	os.makedirs(auxdir)

# ----------------------------------------------------------------
# setup models

# [lam, b, c]
def L_LGM(par=[107, 1/40, 1]):
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

# ----------------------------------------------------------------
# implement multiprocessing


def dataframe(result_list):
	c_list, results, models = zip(*result_list)
	data = {
		'c' : c_list,
		'bic' : [mm.bic() for mm in models],
		'parameters' : [mm.list_parameters() for mm in models],
		'weights' : [mm.weights for mm in models],
		'ks_stat' : [mm.ks_stat for mm in models],
		'n_iteration' : [mm.n_iteration for mm in models],
		'n_free' : [mm.n_parameters() for mm in models],
		'n_constraints' : [mm.n_constraints for mm in models]
	}

	return pd.DataFrame(data)

def ks_test(rvs, construct):
	xlim = rvs.min(), rvs.max()
	xspace = np.linspace(xlim[0], xlim[-1], 2000)
	pdf = construct.pdf(xspace)
	cdf = scipy.integrate.cumulative_trapezoid(pdf, xspace)
	_xspace = (xspace[1:]+xspace[:-1])/2
	f_cdf = scipy.interpolate.interp1d(_xspace, cdf, fill_value=(0,1), bounds_error=False)
	res = scipy.stats.kstest(rvs, f_cdf)
	return res.statistic, res.pvalue


c_bracket = [1,9]
def fit_bracket(data, c_bracket):
	result_list = []
	result_fun = np.inf

	c_min, c_max = c_bracket 
	for c in range(c_min, c_max+1, 1):
		if verbose: print('c = ', c)
		par=[107, 1/40, c]
		construct = L_LGM(par)
		construct.fit(data, tol=1e-5, maxiter=setting_maxiter)
		ks_stat, _ = ks_test(data, construct)
		construct.ks_stat = ks_stat
		# 
		result_list.append([c, construct.result, construct])
		if verbose: print(result_fun , construct.result.fun)
		if exit_early:
			if construct.result.fun < result_fun:
				result_fun = construct.result.fun
			else:
				break # exit early

	return result_list


def test_single(track):

	udata, wavemodel, lptr, meta = get_data(resample(track, n_sample))
	_xlim = (-0.10, np.sqrt(n_sample)*0.10)
	data = fixed_process(udata, xlim=_xlim, remove_zeros=remove_zeros, remove_pairs=remove_pairs)

	# test all models
	with support.PerfTimer() as t:
		result_list = fit_bracket(data, c_bracket)
			
	# timer.append( t.get_time() )
	print('exec time', t.get_time())

	# create a dataframe
	df = dataframe(result_list)

	print('writing to ', 'mdf.pkl')
	df.to_pickle("mdf.pkl")

if preprocess_only:
	sys.exit()

# testing
if test:
	track = track_list[0]
	test_single(track)
	sys.exit()


import multiprocessing

def parallel_run(runner, jobs, maxcores=64):
	running = []

	def check_finished(running):
		still_running = []
		for p in running:
			if p.is_alive():
				still_running.append(p)
		running = still_running
		return running
	
	while(running or jobs):
		while(len(running) >= maxcores):
			running = check_finished(running)
			time.sleep(0.01)
		if jobs:
			job = jobs.pop(0)
			# print('starting job with params {}'.format(job))
			p = multiprocessing.Process(target=runner, args=job)
			p.start()
			running.append(p)
			if not jobs:
				print('waiting for the remaining {} jobs to finish...'.format(len(running)))
		else:
			running = check_finished(running)
			time.sleep(0.01)


def fit_models(data):
	models = []
	timer = []
	for model in model_list:
		mm = model()
		with support.PerfTimer() as t:
			mm.fit(data, tol=1e-4, maxiter=400)
		timer.append( t.get_time() )
		models.append(mm)
	return models, timer

def parallel_compute_em(data_dir, maxcores=8):
	def runner(i, track):
		print("job {} for track {}".format(i, idx_list[i]))

		udata, wavemodel, lptr, meta = get_data(resample(track, n_sample))
		_xlim = (-0.10, np.sqrt(n_sample)*0.10)
		data = fixed_process(udata, xlim=_xlim, remove_zeros=remove_zeros, remove_pairs=remove_pairs)

		# test all models
		with support.PerfTimer() as t:
			result_list = fit_bracket(data, c_bracket)
				
		# timer.append( t.get_time() )
		print('exec time', t.get_time())

		# create a dataframe
		df = dataframe(result_list)

		form = join(data_dir, 'df_{:04d}.pkl')
		target = form.format(idx_list[i])
		print('saving dataframe to ', target)
		df.to_pickle(target)

	jobs = list(enumerate(track_list))
	parallel_run(runner, jobs, maxcores=maxcores)


parallel_compute_em(data_dir, maxcores=args.maxcores)




