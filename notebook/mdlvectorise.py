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
# vectorise the new distance calculation

# %% 
import os
import numpy as np
join = os.path.join 
norm = np.linalg.norm
pi = np.pi

import pili
import _fj

from pili import support
import mdl
import annealing 

# %% 
# load candidate track
candidate_idx = 2924
original_tr = _fj.trackload_original([candidate_idx])[0]
lptr = mdl.get_lptrack(original_tr)
_T = 20
_data = lptr.cut(0,_T)
r = 0.03
M = 10
_guess = mdl.recursive_coarsen(_data, M, parameter='M')
llmodel = annealing.linked_model(_guess)

# %% 
# check that the vectorisation works for n points 
x = _data.get_n2()
# point data
p = x[:10]
ab = _guess.get_n2()
a = ab[:-1]
b = ab[1:]
#
_a = a[0].reshape(1,2)
_b = b[0].reshape(1,2)
support.line_coord_dist(p, _a, _b)

# %% 
def compute_dmatrix():
    for i in range(M-2):
        _a = a[i].reshape(1,2)
        _b = b[i].reshape(1,2)
        support.line_coord_dist(x, _a, _b)

# lets bench mark this
with support.Timer():
    compute_dmatrix()

# %% 
# Jun 14 13:18.  timed 0.06674589s
# Jun 14 14:22   timed 0.00142389s
with support.Timer():
    anneal = annealing.Anneal(r)
    anneal.initialise(_guess, _data)

# %% 
with support.Timer():
    model = list(annealing.nodeiter(llmodel))
    anneal.seglist.update_at(model)

# %% [markdown]
# after reimplementing the distance and s matrices
# multiple_linear_solve was 18x faster
# prority_queue_solve was    8x faster
# ^ I guess the latter could be faster if we write out the O(n) operations

