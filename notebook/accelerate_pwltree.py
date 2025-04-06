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
# we are going to implement an explicit gradient into the fast linear pwl solver
# so use this notebook to benchmark its performance

# %% 

import sys, os
import time
join = os.path.join

import numpy as np

import pili
from pili import support

import matplotlib.pyplot as plt
import matplotlib as mpl

import mdl
import pwlpartition
import pwltree
import pwlstats
import readtrack

# %% 
# need some simulated data to operate on
target = join(pili.root, '../run/825bd8f/target/t0')
track = readtrack.trackset(join(target, 'data/'))[0]

path =  join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
sigma, r = pwlstats.load_candidate_sigma_r(path)

# %% 

dt = np.insert(np.diff(track['time']), 0, 0)
data = mdl.LPtrack(dt, track['x'], track['y'])
short = data.cut(0,500)
len(short)

# %%

with support.PerfTimer() as t:
    wavemodel = pwlpartition.sim_wavemodel(short, sigma)
    solver = pwltree.TreeSolver(short, overlap=True, jac=False)
    solver.build_initial_tree(wavemodel)
    solver.build_priority()
    solver.solve(pwltree.stop_at(r))

# %%
print(f'solved in {t.get_time()}s')
short_model = solver.get_model()
wavemodel = pwlpartition.sim_wavemodel(short, sigma)
print('initial M = ', wavemodel.M)
print('solved M = ', short_model.M)


model = short_model
fig, ax = plt.subplots(figsize=(12,5))
pwlpartition.simple_model_plot(ax, model, short)
ax.plot(model.x, model.y, marker='d')


# %%

with support.PerfTimer() as t:
    wavemodel = pwlpartition.sim_wavemodel(short, sigma)
    jacsolver = pwltree.TreeSolver(short, overlap=True, jac=True)
    jacsolver.build_initial_tree(wavemodel)
    jacsolver.build_priority()
    jacsolver.solve(pwltree.stop_at(r))

# %%

print(f'solved in {t.get_time()}s')
short_model = jacsolver.get_model()
print('initial M = ', wavemodel.M)
print('solved M = ', short_model.M)


# %%
len(jacsolver.history)

model = short_model
fig, ax = plt.subplots(figsize=(12,5))
pwlpartition.simple_model_plot(ax, model, short)
ax.plot(model.x, model.y, marker='d')


# %%
hloss = [h['loss'] for h in solver.get_history()]
jachloss = [h['loss'] for h in jacsolver.get_history()]
ax = plt.gca()
ax.plot(hloss, label='auto')
ax.plot(jachloss, label='jac')
ax.legend()

# %%
# [markdown]
# approx 2x speedup ~ can do better by avoiding repeated caculcations
