
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

single = False
write_auxillary = False

data_dir = join('./', 'test_preprocessing/')
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
	data = emanalyse.symprocess(data, q)
	return data

qlist = [0.005, 0.01, 0.02]
delta_list = [0.005, 0.01]

permutations = [(a, b) for a in qlist for b in delta_list]
# datalist = [preprocess(udata, q) for udata in udatalist]

if write_auxillary:
	# write these data
	auxdir = join('./test_preprocessing/', 'aux/')
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

if preprocessing_only:
	print('finished preprocessing. exit.')
	sys.exit()



# ----------------------------------------------------------------
# setup models

# defaults
_sigma = 0.009
_lam= 40.0

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


# ----------------------------------------------------------------
# implement multiprocessing

def dataframe(models):
	model_names = ['l_eml_fixed' for _ in permutations]
	qs = [ab[0] for ab in permutations]
	deltas = [ab[1] for ab in permutations]
	data = {
		'names' : model_names,
		'q' : qs,
		'delta' : deltas,
		'bic' : [mm.bic() for mm in models],
		'parameters' : [mm.list_parameters() for mm in models],
		'weights' : [mm.weights for mm in models],
		'n_iteration' : [mm.n_iteration for mm in models]
	}

	return pd.DataFrame(data)

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


def fit_models(udata):
	models = []
	timer = []
	for ab in permutations:
		q, delta = ab
		data = preprocess(udata, q, delta)
		mm = L_eml_fixed()
		with support.PerfTimer() as t:
			mm.fit(data, tol=1e-4, maxiter=400)
		timer.append( t.get_time() )
		models.append(mm)
	return models, timer

def parallel_compute_em(data_dir, maxcores=8):
	def runner(i, udata):
		print("job {} for track {}".format(i, idx_list[i]))
		models, timer  = fit_models(udata)
		df = dataframe(models)
		df["exec"] = timer
		form = join(data_dir, 'df_{:04d}.pkl')
		target = form.format(idx_list[i])
		print('saving dataframe to ', target)
		df.to_pickle(target)

	jobs = list(enumerate(udatalist))
	parallel_run(runner, jobs, maxcores=maxcores)


parallel_compute_em(data_dir, maxcores=8)




