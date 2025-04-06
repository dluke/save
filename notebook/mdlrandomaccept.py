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
# test random accept

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

from pili import support
import mdl
import pwlpartition
import synthetic

# %% 
notename = 'mdlrandomaccept'
mplstyle = {"font.size": 24}

def local_plot(model, data, fs=20):
    fig, ax = plt.subplots(figsize=(fs, fs))
    _config = {'match_data': True}
    pwlpartition.plot_wavemodel_on_data(ax, model, data, config=_config)


def mdl_plot(solver, data, fs=20):
    partition = solver.partition
    is_outlier = solver.get_outliers()
    fig, ax = plt.subplots(figsize=(fs, fs))
    _config = {'h_outlier': True}
    mdl.plot_model_on_data(ax, partition.model, data, intermediate={
                           'is_outlier': is_outlier}, config=_config)


# %% 
N = 10
params = {'sigma': 0.10}
pwl, synthdata = synthetic.new_ideal_synthetic(N, params)

# forcible generate a poor initial guess
params = {'coarsen_sigma': 0.5}
wavemodel, data, meta = pwlpartition.initial_guess(synthdata.x, synthdata.y, params)

# %% 
fig, ax = plt.subplots(figsize=(10,10))
pwlpartition.simple_model_plot(ax, wavemodel, data, pwl)


# %% 
# * -------------------------------------------------------------------------------------
# * random_accept
# loss_conf = {'contour_term': 0.00, 'continuity_term': 0}
loss_conf = {}

partition = pwlpartition.PartAnneal(wavemodel, synthdata, loss_conf=loss_conf, 
    use_alternative=True, inter_term=True)
solver = pwlpartition.Solver(partition, r=meta['r'], sigma=meta['sigma'], 
    min_constraint=1, use_description_length=True)
    
solver.linear_solve()
solver.percolate()

# %% 

control = {'maxiter': 30, 't_end' : 0.01, 'tolerance': 1e-8, 'greedy':True}
with support.Timer() as t:
    solver.priority_solve(control)
    
# %% 
mdl_plot(solver, data)
local_plot(solver.partition.model, data)

# %% 
pwlpartition.plot_solver_convergence(solver, labels=['DL', 'M'])

# %% 
# * -------------------------------------------------------------------------------------
# * now do it again but not greedy
loss_conf = {}

partition = pwlpartition.PartAnneal(wavemodel, synthdata, loss_conf=loss_conf, 
    use_alternative=True, inter_term=True)
solver = pwlpartition.Solver(partition, r=meta['r'], sigma=meta['sigma'], 
    min_constraint=1, use_description_length=True)
    
solver.linear_solve()

# %% 

control = {'maxiter': 30, 't_end' : 0.01, 'tolerance': 1e-8, 'greedy':False}
with support.Timer() as t:
    solver.priority_solve(control)
    
# %% 
mdl_plot(solver, data)
local_plot(solver.partition.model, data)

# %% 
M = solver.partition.model.M
thermostat = solver.get_thermostat()
thermostat.tlist[0] = [0.2,0.5,0.1]
thermostat.diffuse_temperature_at(0)
# score, loss = solver.get_history()
pwlpartition.plot_solver_convergence(solver, labels=['DL', 'M'])


# %% 
# partition = solver.partition
# M = solver.partition.model.M
# partition.inter_residual_at(0, M)

# partition.model.get_n2()[0], partition.data.get_n2()[0]
# partition.residual_at(0, M)
# partition.model.dt

# %% 

# solver.partition.model.dt


# %% 

# fig, ax = plt.subplots(figsize=(10,10))
# pwlpartition.simple_model_plot(ax, solver.partition.model, solver.partition.data, pwl)

solver.dump_state('mdlpartition/greedy_accept')

