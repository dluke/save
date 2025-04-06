
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
import _fj


# ----------------------------------------------------------------
# preprocessing the data

# analyse the crawling dataset
# * LOAD the full dataset


data_dir = join('./', 'preprocessing/')
if not os.path.exists(data_dir):
	print('create directory ', data_dir)
	os.makedirs(data_dir)

print('using directory ', data_dir)

# smooth
wavelet='db4'
em_config = {"wavelet":wavelet, 'method':'VisuShrink', "mode":'soft', "rescale_sigma":False}

with open(join(data_dir, "config.json"), 'w') as f:
	json.dump(em_config, f)

load_path = join(pili.root, 'notebook/thesis_classification/crawling.npy')
idx_list = np.loadtxt(load_path).astype(int)
track_list = _fj.trackload_original(idx_list)

print("Loaded {:d} trajectories from {}".format(len(idx_list), load_path))

def get_data(track):
	wavemodel, lptr, meta = emanalyse.local_wave_guess(track, config=em_config)
	curve_coord = emanalyse.compute_curve_coordinate(wavemodel, lptr, adjacent=True, ln=101)
	udata = np.diff(curve_coord) 
	return udata, wavemodel, lptr, meta

for i, track in enumerate(track_list):
	udata, wavemodel, lptr, meta = get_data(track) 
	out = join(data_dir, 'udata_{:04d}.pkl'.format(i))
	with open(out, 'wb') as f:
		print('writing to ', out)
		pickle.dump(udata, f)


