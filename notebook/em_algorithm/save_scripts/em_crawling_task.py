
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
import _fj
import pwlpartition

# ----------------------------------------------------------------
# control flags

preprocessing_only = True

test = False
single = False

data_dir = join('./', 'crawling_em_aux/')
if not os.path.exists(data_dir):
	print('create directory ', data_dir)
	os.makedirs(data_dir)

# ----------------------------------------------------------------
# preprocessing the data

# analyse the crawling dataset
# * LOAD the full dataset

wavelet='db1'
em_config = {"wavelet":wavelet, 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}

load_path = join(pili.root, 'notebook/thesis_classification/crawling.npy')
idx_list = np.loadtxt(load_path).astype(int)
if single: 
	idx_list = idx_list[:1]
track_list = _fj.trackload_original(idx_list)

print("Loaded {:d} trajectories from {}".format(len(idx_list), load_path))

def get_data(track):
	wavemodel, lptr, meta = pwlpartition.wavelet_guess(track, config=em_config)
	curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr)
	udata = np.diff(curve_coord) 
	return udata, wavemodel, lptr, meta

print("Estimate the trajectory shape and perform local mapping.")
model_list = [get_data(track) for track in tqdm(track_list)]
udatalist = [m[0] for m in model_list]
wavelist = [m[1] for m in model_list]
lptrlist = [m[2] for m in model_list]
metalist = [m[3] for m in model_list]

# important meta parameter controlling the data preprocessing
def preprocess(data, q=0.01):
	data = data[np.isfinite(data)]
	data = emanalyse.remove_pairs(data)
	data = emanalyse.symprocess(data, q)
	return data

q = 0.01
datalist = [preprocess(udata, q) for udata in udatalist]



# write these data
auxdir = join('./', 'aux/')
if not os.path.exists(auxdir):
	print('create directory ', auxdir)
	os.makedirs(auxdir)

# write these data

suffix = 'crawling_'
p1 = join(auxdir, suffix + 'udatalist.pkl')
with open(p1, 'wb') as f:
	print('writing to ', p1)
	pickle.dump(udatalist, f)

p2 = join(auxdir, suffix + 'metalist.pkl')
with open(p2, 'wb') as f:
	print('writing to ', p2)
	pickle.dump(metalist, f)

p3 = join(auxdir, suffix + 'wavelist.pkl')
with open(p3, 'wb') as f:
	print('writing to ', p3)
	pickle.dump(wavelist, f)


p4 = join(auxdir, suffix + 'lptrlist.pkl')
with open(p4, 'wb') as f:
	print('writing to ', p4)
	pickle.dump(lptrlist, f)

p5 = join(auxdir, suffix + 'datalist.pkl')
with open(p5, 'wb') as f:
	print('writing to ', p5)
	pickle.dump(datalist, f)

if preprocessing_only:
	print('finished preprocessing. exit.')
	sys.exit()



# ----------------------------------------------------------------
# setup models

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
	return construct

def g_emg_constraint_m0(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, numerical=True, n_constraints=1, sigma_constraint=True)
	construct.add_impulse(0, fix_loc=False)
	construct.add_exponential(0, lam)
	return construct
	
def g_emg_fixed(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, numerical=True, n_constraints=1, sigma_constraint=True)
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0, lam, fix_loc=True)
	return construct

def g_emg(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, numerical=True, n_constraints=0, sigma_constraint=False)
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0, lam)
	return construct


def L_emg(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, 
		numerical=True, n_constraints=0, sigma_constraint=False, err='laplace')
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0, lam)
	return construct

def L_emg_fixed(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, 
		numerical=True, n_constraints=0, sigma_constraint=False, err='laplace')
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0, lam, fix_loc=True)
	return construct


def L_eml(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, 
		numerical=True, n_constraints=1, sigma_constraint=False, err='laplace')
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0.01, None, err='laplace')
	return construct

def L_eml_fixed(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, 
		numerical=True, n_constraints=1, sigma_constraint=False, err='laplace')
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0.0, None, err='laplace', fix_loc=True)
	return construct


model_names = ['gmm2', 'gmm2_constraint', 'g_emg', 'g_emg_constraint', 'g_emg_constraint_m0', 'gmm3']
model_list = [gmm2, gmm2_constraint, g_emg, g_emg_constraint, g_emg_constraint_m0, gmm3]

model_names = ['g_emg', 'g_emg_fixed', 'L_emg_fixed', 'L_eml', 'L_eml_fixed']
model_list = [g_emg, g_emg_fixed, L_emg_fixed, L_eml, L_eml_fixed]


# ----------------------------------------------------------------
# implement multiprocessing

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

# testing
if test:
	data = datalist[0]
	test_single(data)

def test_single(data):
	# test all models
	timer = []
	with support.Timer():
		models = []
		for model in model_list:
			mm = model()
			with support.PerfTimer() as t:
				mm.fit(data, tol=1e-4, maxiter=400)
			timer.append( t.get_time() )
			models.append(mm)

	# create a dataframe
	df = dataframe(models)
	df["exec"] = timer

	df.to_pickle("df.pkl")


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
	def runner(i, data):
		print("job {} for track {}".format(i, idx_list[i]))
		models, timer  = fit_models(data)
		df = dataframe(models)
		df["exec"] = timer
		form = join(data_dir, 'df_{:04d}.pkl')
		target = form.format(idx_list[i])
		print('saving dataframe to ', target)
		df.to_pickle(target)

	jobs = list(enumerate(datalist))
	parallel_run(runner, jobs, maxcores=maxcores)


parallel_compute_em(data_dir, maxcores=8)




