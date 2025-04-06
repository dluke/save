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
# our PWL solver  is working somewhat well
# we can tune the control parameters in order to do better
# in particular we would like to explore uphill in the search space before looking for the minimum

# %% 
import os
from copy import deepcopy
import random
import numpy as np
import pickle
join = os.path.join 
norm = np.linalg.norm
pi = np.pi

import scipy
import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
import pili

import _fj
print("writing figures to", pub.writedir)

from pili import support
import mdl
import annealing 

# %%
def local_plot(anneal, data, _config={"h_outlier":True}, fsize=20):
    # note specific plotting function
    fig, ax = plt.subplots(figsize=(fsize,fsize))
    is_outlier = anneal.get_outliers()
    mdl.plot_model_on_data(ax, anneal.get_current_model(), data, 
        intermediate={'is_outlier':is_outlier}, config=_config)


# %% 
# load candidate track
r = 0.03
candidate_idx = 2924
original_tr = _fj.trackload_original([candidate_idx])[0]
lptr = mdl.get_lptrack(original_tr)

i = 7
T = 20
Ti = i * T + 10
Tf = (i+1) * T + 10
_data = lptr.cut(Ti, Tf)
M = int(len(_data)/10) + 1 - 10
_guess = mdl.recursive_coarsen(_data, M, parameter='M')

fig, ax = plt.subplots(figsize=(20,20))
mdl.plot_model_on_data(ax, _guess, _data)

# %% 

# setup solver
chunk_anneal = annealing.Anneal(r)
chunk_anneal.initialise(_guess, _data)
rng = np.random.RandomState(0)
chunk_solver = annealing.Solver(chunk_anneal, rng=rng)

# loss function configuration
loss_conf = {}
loss_conf["contour_term"] = 0.01
# loss_conf["debug"] = True
chunk_solver.anneal.default_loss_conf = loss_conf

pqsolve_config = {
    'greedy' : False, 
    'Tc' : 0.01, 
    'fc' : 0.5
    }

#
chunk_solver.multiple_linear_solve()
chunk_solver.priority_solve(control=pqsolve_config)

# %% 
update_config = pqsolve_config.copy()
update_config['greedy'] = True
update_config['fc'] = 0.2
chunk_solver.priority_solve(control=update_config)

# %% 
chunk_solver.remapping()


# %% 
local_plot(chunk_solver.anneal, _data)

# %% 
# ----------------------------------------------------------------      
# picking a  new track to operate on, this time from the bottom of the top 200 crawling trajectories
trackidx = 1687

original_tr = _fj.trackload_original([trackidx])[0]
lptr = mdl.get_lptrack(original_tr)

i = 0
T = 20
Ti = i * T 
Tf = (i+1) * T 
_data = lptr.cut(Ti, Tf)
M = int(len(_data)/20) + 1
_guess = mdl.recursive_coarsen(_data, M, parameter='M')

fsize = (10,10)
fig, ax = plt.subplots(figsize=fsize)
mdl.plot_model_on_data(ax, _guess, _data)

# %% 
# define the solver

def solve_path(solver):
    # loss function configuration
    loss_conf = {}
    loss_conf["contour_term"] = 0.01
    # loss_conf["debug"] = True
    solver.anneal.default_loss_conf = loss_conf

    pqsolve_config = {
        'greedy' : False, 
        'Tc' : 0.01, 
        'fc' : 0.5
        }

    #
    solver.multiple_linear_solve()
    solver.priority_solve(control=pqsolve_config)

    update_config = pqsolve_config.copy()
    update_config['greedy'] = True
    update_config['fc'] = 0.2
    solver.priority_solve(control=update_config)

    solver.cleanup()
    
    return solver

# %% 

chunk_anneal = annealing.Anneal(r)
chunk_anneal.initialise(_guess, _data)
rng = np.random.RandomState(0)
chunk_solver = annealing.Solver(chunk_anneal, rng=rng)

    
chunk_solver = solve_path(chunk_solver)

# %% 

T = 20
n_chunk = int(lptr.get_duration()/T)
save_chunk = []
save_data = []
for i in range(n_chunk):
    print("----------------------------------------------------------------")
    print(f"chunk {i}")
    Ti = i * T 
    Tf = (i+1) * T 
    _data = lptr.cut(Ti, Tf)
    M = int(len(_data)/20) + 1
    _guess = mdl.recursive_coarsen(_data, M, parameter='M')

    chunk_anneal = annealing.Anneal(r)
    chunk_anneal.set_mapping_constraint(False)
    chunk_anneal.initialise(_guess, _data)
    chunk_solver = annealing.Solver(chunk_anneal, rng=np.random.RandomState(0))

    solve_path(chunk_solver)

    save_data.append(_data)
    save_chunk.append(chunk_solver.anneal)

print("Finished")


# %% 
# plot chunks
for i in range(len(save_chunk)):
    local_plot(save_chunk[i], save_data[i], fsize=10)
    plt.gca().set_title(f'chunk {i}')


# %% 
# solve the whole thing 
# (turn off mapping constriant for speed) #todo check minargmin is correct
_data = lptr
M = int(len(_data)/20) + 1
print("initial guess M = ", M)
_guess = mdl.recursive_coarsen(_data, M, parameter='M')

anneal = annealing.Anneal(r)
anneal.initialise(_guess, _data)
anneal.set_mapping_constraint(False) #!
solver = annealing.Solver(anneal, rng=np.random.RandomState(0))

with support.Timer() as t:
    solve_path(solver)


# %% 


local_plot(solver.anneal, _data,  fsize=100)

# %% 

path = join(pili.root, "../sparseml", "run/top/notebook", f"track_{trackidx}")
solver.dump_state(path)


