
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

preprocessing_only = False

test = False
single = False

data_dir = join('./', 'crawling_em_zfix/')
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
mlist = [get_data(track) for track in tqdm(track_list)]
udatalist = [m[0] for m in mlist]
wavelist = [m[1] for m in mlist]
lptrlist = [m[2] for m in mlist]
metalist = [m[3] for m in mlist]

# important meta parameter controlling the data preprocessing
def preprocess(data, q=0.005, delta=0.005):
	data = data[np.isfinite(data)]
	data = emanalyse.remove_pairs(data ,delta=delta)
	data = emanalyse.asymprocess(data, q)
	return data

q = 0.005
delta = 0.005
datalist = [preprocess(udata, q, delta) for udata in udatalist]


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
	construct.add_impulse(0.01)
	return construct

def gmm2_constraint(sigma=_sigma):
	construct = MixtureModel(sigma, numerical=True, n_constraints=1, sigma_constraint=True)
	construct.add_impulse(0, fix_loc=True)
	construct.add_impulse(0.01)
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
	construct.add_exponential(0.01, lam)
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

def L_eml(sigma=_sigma, lam=_lam):
	construct = MixtureModel(sigma, 
		numerical=True, n_constraints=1, sigma_constraint=False, err='laplace')
	construct.add_impulse(0, fix_loc=True)
	construct.add_exponential(0.01, None, err='laplace')
	def pack(mix):
		return np.array([mix[1].l1, mix[1].loc, mix[1].l2])
	def unpack(self, x):
		l1, m1, l2 = x
		self.mix[0].scale = 1/l2
		self.mix[1].loc = m1
		self.mix[1].l1 = l1
		self.mix[1].l2 = l2
	construct.set_numerical_instructions(pack, unpack)
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
	construct.add_impulse(0, fix_loc=True)
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

model_names = ['gmm2', 'g_emg_constraint', 'g_emg_fixed', 'L_eml', 'L_eml_fixed', 'L_NL']
model_list = [gmm2, g_emg_constraint, g_emg_fixed, L_eml, L_eml_fixed, L_NL]
print('model_list', model_names)

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

	print('writing to ', 'df.pkl')
	df.to_pickle("df.pkl")

# testing
if test:
	data = datalist[0]
	test_single(data)
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




