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
# plot the curve coordinate (mapped time)
# we want to know if we can see and define "tumbles" in the mapped time

# %% 
import os
import random
import numpy as np
import pickle
import collections
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
from synthetic import *


# %%
def local_plot(anneal, data, _config={"h_outlier":True}, fsize=20):
    # note specific plotting function
    fig, ax = plt.subplots(figsize=(fsize,fsize))
    is_outlier = anneal.get_outliers()
    mdl.plot_model_on_data(ax, anneal.get_current_model(), data, 
        intermediate={'is_outlier':is_outlier}, config=_config)


# %% 
# load candidate track
candidate_idx = 2924
original_tr = _fj.trackload_original([candidate_idx])[0]
lptr = mdl.get_lptrack(original_tr)
# _data = lptr.cut(0,40)
_data = lptr
r = 0.03
print("data has {} points".format(len(_data.x)))
# %% 
M = int(len(_data)/10) + 1
_guess = mdl.recursive_coarsen(_data, M, parameter='M')
ax = plt.subplots(figsize=(20, 8))
mdl.plot_model_on_data(plt.gca(), _guess, _data)

# %% 
# config
M = int(len(_data)/10) + 1
print("initial guess M = ", M)
config = {"use_ordering": True, "greedy": True}
rngstate = np.random.RandomState(0)


_guess = mdl.recursive_coarsen(_data, M, parameter='M')
# anneal = annealing.Anneal(r, k=1, use_description_length=False)
anneal = annealing.Anneal(r)
anneal.initialise(_guess, _data)

solver = annealing.Solver(anneal, rng=rngstate, config=config)

annealing._debug = False
# annealing.f = open('annealing/out.txt', 'w') 
annealing.f = sys.stdout

with support.Timer():
    solver.multiple_linear_solve(n=1, n_local=2)
solver.dump_state("candidate_guess")

# %%
solver = annealing.Solver.load_state("candidate_guess")
annealing._debug = True
with support.Timer():
    solver.priority_solve()
solver.dump_state("candidate")

# %%
print(solver.anneal.seglist.closest_segment)
sidx = solver.anneal.get_current_model().sidx[:-1]
for s in sidx:
    print(s, sum(solver.anneal.seglist.closest_segment == s))

# %%
solver.remapping()


# %%

local_plot(solver.anneal, _data, _config={"h_outlier":True}, fsize=200)

# %%
# _data = lptr.cut(0,40)
# debugsolver = annealing.Solver.load_state("partial")
_data = lptr
debugsolver = annealing.Solver.load_state("/home/dan/usb_twitching/sparseml/run/partial")

# %%

local_plot(debugsolver.anneal, _data, _config={"h_outlier":True})
debugsolver.anneal.llmodel


# %%
# --------------------------------
# implement remapping solver
solver = annealing.Solver.load_state("candidate_t20")
local_plot(solver.anneal, _data, _config={"h_outlier":True})

# %%

rng = np.random.RandomState()
solver.rng = rng
solver.remapping()
print('finished')
local_plot(solver.anneal, _data, _config={"h_outlier":True})

# %% [markdown]
# it actually works. 

# %%

_T = 40
_data = lptr.cut(0,_T)
r = 0.03
M = 21
_guess = mdl.recursive_coarsen(_data, M, parameter='M')
ax = plt.subplots(figsize=(20, 8))
mdl.plot_model_on_data(plt.gca(), _guess, _data)



# %%

def solve_exp(_data, M, r):
    config = {"use_ordering": True}
    rngstate = np.random.RandomState(0)
    _guess = mdl.recursive_coarsen(_data, M, parameter='M')
    anneal = Anneal(r)
    anneal.initialise(_guess, _data)
    solver = annealing.Solver(anneal, rng=rngstate, config=config)
    print("initial solve")
    solver.multiple_linear_solve(n=1, n_local=2)
    print("priority solve")
    solver.priority_solve()
    return solver

solver = solve_exp(_data, M, r)

# %%
local_plot(solver.anneal, _data, _config={"h_outlier":True})
plt.savefig(join(pili.root, "../sparseml/save/candidate_40.svg"))

# %%
topidx = _fj.load_subset_idx()["top"][0]
toptrack = _fj.trackload_original([topidx])[0]
toplptr = mdl.get_lptrack(toptrack)
# fig, ax = plt.subplots(figsize=(40,40))
mdl.plot_model_on_data(plt.gca(), None, toplptr)
plt.savefig(join(pili.root, "../sparseml/save/toptrack.svg"))

# %%
_T = 20
top_data = toplptr.cut(0,_T)
fig, ax = plt.subplots(figsize=(10,10))
mdl.plot_model_on_data(plt.gca(), None, top_data)

# %%
M = 11
_guess = mdl.recursive_coarsen(top_data, M, parameter='M')
ax = plt.subplots(figsize=(10,10))
mdl.plot_model_on_data(plt.gca(), _guess, top_data)

# %%
solver = solve_exp(top_data, M, r)
# %%
local_plot(solver.anneal, top_data, _config={"h_outlier":True}, fsize=10)


# %% [markdown]
# average displacement
# candidate: 0.399
# top: 0.242

