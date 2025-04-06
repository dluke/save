
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
import _fj
import mdl
import pwlpartition
import pwltree

# ----------------------------------------------------------------
# control flags

test = False
single = False

data_dir = join('./', 'annealing/')
if not os.path.exists(data_dir):
	print('create directory ', data_dir)
	os.makedirs(data_dir)

print('using directory ', data_dir)


# ----------------------------------------------------------------
# * LOAD the full dataset

load_path = join(pili.root, 'notebook/thesis_classification/crawling.npy')
idx_list = np.loadtxt(load_path).astype(int)
if single: 
	idx_list = [idx_list[650]]
track_list = _fj.trackload_original(idx_list)


print("Loaded {:d} trajectories from {}".format(len(idx_list), load_path))


# work functions

def pwl_tree_solve(data, r):
	tsolver = pwltree.TreeSolver(data, overlap=True)
	tsolver.build_max_tree()
	tsolver.build_priority()
	tsolver.solve(pwltree.stop_at(r))
	return tsolver

def get_lptr(track):
	dt = np.insert(np.diff(track['time']), 0, 0)
	lptr = mdl.LPtrack(dt, track['x'], track['y'])
	return lptr

def estimate_sigma(track):
	x, y = track['x'], track['y']
	return pwlpartition.estimate_error(x, y)



# compute
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

fixed_sigma = 0.009249664147015167


def tree_process(track, shorten=None):

	sigma = pwlpartition.estimate_error(track)
	r_stop = pwlpartition.estimate_r(sigma)
	if shorten != None:
		track = track.cut(shorten)

	# work
	lptr = get_lptr(track)
	solver = pwl_tree_solve(lptr, r_stop)
	return solver

def pwl_annealing(i, track, shorten=None):

	# compute 
	lptr = mdl.get_lptrack(track)
	_, _, meta = pwlpartition.initial_guess(lptr.x, lptr.y)

	if shorten != None:
		track = track.cut(0,shorten)

	lptr = mdl.get_lptrack(track)
	wavemodel, lptr, _ = pwlpartition.initial_guess(lptr.x, lptr.y)

	loss_conf = {}
	partition = pwlpartition.PartAnneal(wavemodel, lptr, loss_conf=loss_conf, inter_term=True)
	rng = np.random.RandomState()
	solver = pwlpartition.Solver(partition, rng=rng, r=meta["r"], sigma=meta["sigma"], min_constraint=1)
	control = {'maxiter': 5000, 't_end': 0., 'tolerance': 1e-6}

	with support.PerfTimer() as timer:
		solver.linear_solve()
		solver.percolate()

	# output = pwlpartition.Output(target_dir, allow_write=True, index=trackidx)
	with support.PerfTimer() as timer:
		solver.priority_solve(control, output=None)

	solver.track_index = idx_list[i]
	solver.exec_time = timer.get_time()

	return solver
	

def parallel_compute_em(data_dir, track_list, maxcores=8, shorten=None):
	form = "pwl_{:04d}.pkl"
	def runner(i, track):
		print("job {} for track {}".format(i, idx_list[i]))

		# solver = tree_process(track, shorten)
		solver = pwl_annealing(i, track, shorten)

		# output
		target = join(data_dir, form.format(idx_list[i]))
		print('saving data to ', target)
		with open(target, 'wb') as f:
			pickle.dump(solver, f)


	jobs = list(enumerate(track_list))
	parallel_run(runner, jobs, maxcores=maxcores)


shorten = 1000

if test:
	# short_data = track_list[0].cut(0,800)
	parallel_compute_em(data_dir, [track_list[0]], shorten=shorten)
	sys.exit()

parallel_compute_em(data_dir, track_list, maxcores=8, shorten=shorten)




