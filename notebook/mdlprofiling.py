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
# profile our methods on ideal data
# use standard python profiling tools and also analyse niter/nfev for local optimisation

# %% 
import os
import random
import numpy as np
import pickle
join = os.path.join 
norm = np.linalg.norm
pi = np.pi

import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili
from pili import support
import mdl
import pwlpartition
import synthetic

# %% 
notename = 'mdlprofiling'
mplstyle = {"font.size": 24}

# %% 
# profiling and existing solver
target = join(pili.root, "notebook/mdlpartition/shortchi2")
solver = pwlpartition.Solver.load_state(target)

def profile(solver):
    resls = solver.partition._cache_local_optimise_result
    local_solve_time = [res.exec_time for res in resls]
    nfev = [res.nfev for res in resls]

    # print(local_solve_time)
    print(len(nfev))


profile(solver)
# pwlpartition.plot_solver_convergence(solver, labels=['k', 'M', 'P(M)'])

# %% 
# construct short synthetic data that we can rapidly profile
np.random.seed(0)

def new_ideal_synthetic(N):
    _l = 1.0
    sigma = 0.05
    length = synthetic.Uniform(_l, _l)

    dx = synthetic.Constant(_l/10)
    error = synthetic.Normal(scale=sigma)

    # test mirrored normal 
    mnormal = synthetic.AlternatingMirrorNormal(loc=pi/4, scale=pi/16)
    
    pwl = synthetic.new_static_process(length, mnormal, N)
    synthdata = synthetic.sample_pwl(pwl, dx, error)
    return pwl, synthdata

N = 5
pwl, synthdata = new_ideal_synthetic(N)

def prep(synthdata):
    wavemodel, data, meta = pwlpartition.initial_guess(synthdata.x, synthdata.y)
    loss_conf = {}
    partition = pwlpartition.PartAnneal(wavemodel, synthdata, loss_conf={}, inter_term=True)
    solver = pwlpartition.Solver(partition, r=meta['r'], sigma=meta['sigma'], 
        min_constraint=1, use_description_length=True)
    return solver

solver = prep(synthdata)

fig, ax = plt.subplots(figsize=(10,10))
pwlpartition.simple_model_plot(ax, solver.partition.model, solver.partition.data, pwl)

# %% 
import cProfile

control = {'maxiter': 100, 'tolerance': 1e-8, 'greedy':False}

with cProfile.Profile() as pr:
    solver.priority_solve(control)

pr.print_stats()

fig, ax = plt.subplots(figsize=(10,10))
pwlpartition.simple_model_plot(ax, solver.partition.model, solver.partition.data, pwl)

# %% 
import pstats
p = pstats.Stats(pr)
p.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(50)
