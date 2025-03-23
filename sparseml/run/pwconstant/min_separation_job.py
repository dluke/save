
import os
import numpy as np
join = lambda *x: os.path.abspath(os.path.join(*x))

from pwcsearch import *
import filesystem

#! output
data_directory = "min_separation/"
filesystem.safemkdir(data_directory)

param = {
	'sigma': 0.2,
	'min_separation': 0.3,
	'err_lambda': 2
	}

# new random state
random_state = np.random.default_rng()

N = 500

#! make_jobs

var_min_separation = np.linspace(0.0, 0.5, 6, True)
variant = ('min_separation', np.linspace(0.0, 0.5, 6, True))

if __name__=='__main__':
	parallel_search(variant, N, param, data_directory)


