
# %% [markdown]
# implement a deterministic solver based on a priority queue



# %% 
import os
import random
import numpy as np
join = lambda *x: os.path.abspath(os.path.join(*x))

from copy import copy, deepcopy

import scipy
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import thesis.publication as thesis

import pili
from pili import support
import mdl
import pwltree

from pwcsearch import *

images_dir = join(pili.root, "../sparseml/images")

# %% 
# rng = np.random.default_rng(0)

d = []
for i in range(10000):

	# _y = y[-2]
	# _r =  shuffle.get_residual_stack()[-1]
	_r = np.abs(0.5 * rng.standard_normal(60))
	# print(_r)

	x = np.sum( (_r/0.5)**2 )

	p = scipy.stats.chi2.cdf(x, _r.size)
	d.append(p)

sns.histplot(d)
