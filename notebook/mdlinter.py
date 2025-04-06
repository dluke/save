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
# test overlapping the piecewise data by one data point
# can pull attempted solves from 
# /home/dan/usb_twitching/sparseml/run/synthetic/inter_vary_sigma

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
import pwlstats

# %% 
notename = 'mdlinter'
mplstyle = {"font.size": 24}

# %% 
target = join(pwlstats.root, "run/synthetic/save/inter_vary_sigma/sigma_00.1000")
solver = pwlpartition.Solver.load_state(join(target, "solver"))

# %% 
fig, ax = pwlpartition.model_plot(solver, solver.partition.data)
ax.set_xlim((5,15))

# %% 
# cut the intersting part
time = solver.partition.model.get_time()
si, sf = 8, 20
part = solver.partition.model.cut(time[si], time[sf])
partdata = solver.partition.data.cut_index(time[si], time[sf])
part.dt[0] = 0
print(solver.partition.model.dt)

def local_plot(part, partdata):
    fig, ax = plt.subplots(figsize=(10,10))
    pwlpartition.simple_model_plot(ax, part, partdata)
local_plot(part, partdata)

# %% 
# try again to solve this chunk
partition = pwlpartition.PartAnneal(part, partdata, loss_conf=loss_conf, 
    use_alternative=False, inter_term=True, use_bounds=True)
_solver = pwlpartition.Solver(partition, r=r, sigma=sigma, min_constraint=1)

control = {'maxiter': 200, 't_end': 0., 'tolerance': 1e-8}

with support.Timer() as t:
    _solver.priority_solve(control, output=None)

# %% 
local_plot(_solver.partition.model, _solver.partition.data)

# %% 
msolver = _solver.clone()
print('score', msolver.get_score())
msolver.create_at(4)
print('score', msolver.get_score())
local_plot(msolver.partition.model, msolver.partition.data)

# %% 
msolver = _solver.clone()
print('score', msolver.get_score())
print('start', msolver.partition.get_total_loss())
msolver.partition.create_at(4)
print('create', msolver.partition.get_total_loss())
msolver.partition.local_optimise_at(2, 7)
print('local', msolver.partition.get_total_loss())

msolver.partition = msolver.binary_percolate_at(4, partition=msolver.partition)

print()
# msolver.partition = msolver.percolate_at(4)

# msolver.partition = msolver.percolate_one(4, pwlpartition.REVERSE)

# print(msolver.partition.get_total_loss())
print('percolate', msolver.partition.get_total_loss())
print('score', msolver.get_score())

local_plot(msolver.partition.model, msolver.partition.data)

# %% 
i, f = 0, _solver.partition.model.M-1
_solver.partition.residual_at(i, f)
_solver.partition.inter_residual_at(i, f)
