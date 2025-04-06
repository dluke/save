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
# examine simulated data again but this time compare simulation against pwl solve of candidate trajectory
# we will not compare pwl solves of the simulated data here just yet, see mdlpwltmos.py work in that direction

# %%
import os
import json
import numpy as np
import scipy.stats
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm

import readtrack
import parameters
import _fj

from pili import support
import pili

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import mdl
import pwlpartition
import pwlstats


# %%
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
notename = 'mdltmos'
mplstyle = {"font.size": 20}

# %%
# load simulation
simid = 'HH161KlZ'
target = join(pili.root, '../run/825bd8f/cluster/mc4d_vret/_u_HH161KlZ')
track = readtrack.Track(join(target, 'data/bacterium_00000.dat'))
param = parameters.thisread(directory=target)

xydisp = np.load(join(target, 'xydisp.npy'))


# load candidate
path = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
summary = pwlstats.load_candidate_statistics(path)
sigma = summary['estimate_sigma']
r = pwlpartition.estimate_r(sigma)
meta = {'sigma':sigma, 'r':r}
print(f'candidate sigma = {sigma} r = {r}')

# %%
color = itertools.cycle(['#FEC216', '#F85D66', '#75E76E'])

# %%
# plotting the distributions of candidate pwl model step length
# and per pilus displacement from simulation
# these two items are in a rough sense comparable since we identify pwl segments in the tracking data
# with retraction events
pwl_length = summary['lengths']
# describe
print('entries', xydisp.size, summary['pwlm'])
print('min max', pwl_length.min(), pwl_length.max())
print('min max', xydisp.min(), xydisp.max())
print('P(entry < empirical minimum) = ', np.mean(xydisp < pwl_length.min()))
print('P(entry < 2r) = ', np.mean(xydisp < 2*r))
# ~

fig , ax = plt.subplots(figsize=(6,4))
hstyle = {'stat':'density', 'alpha':0.4, 'element':'step'}
sns.histplot(xydisp, ax=ax, color=defcolor[0], **hstyle)
sns.histplot(pwl_length, ax=ax, color=defcolor[1],  **hstyle)
ax.set_xlabel(r'length $\mu m$')
ax.legend([simid+' pdisp', 'candidate pwl'])

# %%
[param.pget('k_spawn'), param.pget('pilivar'), param.pget('dwell_time'), param.pget('kb_sh') * 0.004]

# %%

data = track
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.3, 'color':defcolor[2]}

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(data['x'], data['y'], **ptlkw)


short = track.cut_time(0, 40)

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(short['x'], short['y'], **ptlkw)

# %%
shortdata = mdl.LPtrack.from_track(short)

wavemodel = pwlpartition.sim_wavemodel(shortdata, sigma)
print('wavemodel.M = ', wavemodel.M)
print('human count M ~ 9')

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(short['x'], short['y'], **ptlkw)

pwlpartition.simple_model_plot(ax, wavemodel, shortdata)
ax.plot(wavemodel.x, wavemodel.y, marker='d')
wavemodel.dt

# %%

# partition = pwlpartition.PartAnneal(wavemodel, shortdata, 
#     use_alternative=True, inter_term=True)
# solver = pwlpartition.Solver(partition, r=r, sigma=sigma, min_constraint=1)

# control = {'maxiter': 300, 't_end': 0., 'tolerance': 1e-8, 'greedy':False}

# with support.Timer() as t:
#     solver.priority_solve(control, output=None)

# pwlpartition.model_plot(solver, shortdata)

# %%
# linear tree
# --- just example syntax ---
from sklearn import tree
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=4,
                       n_informative=2, n_targets=1,
                       random_state=0, shuffle=False)
regr = LinearTreeRegressor(base_estimator=LinearRegression())
regr.fit(X, y)


# %%
from sklearn.tree import export_text
# run the lineartreeregression on real data
X = short['time'][:,np.newaxis]
y = short.get_n2()

# regr = LinearTreeRegressor(base_estimator=LinearRegression())
# regr = tree.DecisionTreeRegressor(max_depth=5)
regr = tree.DecisionTreeRegressor(max_depth=3)
regr.fit(X, y)
text = export_text(regr)
print(text)

# regr.predict([[40]])
base = np.linspace(0,40,100)
x, y = regr.predict([[t] for t in base]).T

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(short['x'], short['y'], **ptlkw)
ax.plot(x,y)


# %%
# test tree method
import pwltree

tsolver = pwltree.TreeSolver(shortdata, overlap=True)
tsolver.build_initial_tree(wavemodel)
tsolver.build_priority()
print([child.future_loss for child in tsolver.root.children])

# tsolver.priority_join()
tsolver.solve(pwltree.stop_at(r))

tsolver.history[-1]

# %%
# for node in tsolver.root.children:
#     print(node.future_loss)

# %%

# def plot_disconnected(ax, lines):
#     for line in lines:
#         _x, _y = zip(*line)
#         ax.plot(_x, _y)
#     ax.set_aspect("equal")

# fig, ax = plt.subplots(figsize=(12,5))
# ax.plot(shortdata.x, shortdata.y, **ptlkw)
# plot_disconnected(ax, lines)

# %%
model = tsolver.get_model()

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1, c2, c3 = defcolors[:3] 
model_style = {"linestyle": '-', 'marker': 'D', 'lw':4, 'alpha':0.5, 'color':c2}
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.3, 'color':c3}

def local_simple_plot(ax, model=None, data=None):
    ax.plot(data.x, data.y, label='data', **ptlkw)
    if model:
        ax.plot(model.x, model.y, label='PWL model', **model_style)
    ax.set_aspect('equal')
    # ax.legend(fontsize=20, loc=(1.04, 0))
    ax.legend(fontsize=20)

fig, ax = plt.subplots(figsize=(12,5))
local_simple_plot(ax, model, shortdata)

plot_target = join(pwlstats.root, "impress/images")
target = join(plot_target, "sim_treesolve_example.png")
plt.tight_layout()
plt.savefig(target)

fig, ax = plt.subplots(figsize=(12,5))
local_simple_plot(ax, None, shortdata)

target = join(plot_target, "sim_treesolve_example_data.png")
plt.tight_layout()
plt.savefig(target)


# %%
steps = model.get_step_length()

def solve_lead_trail(track):
    dt = np.insert(np.diff(track['time']), 0, 0)
    lead_data = mdl.LPtrack(dt, track['x'], track['y'])
    trail_data = mdl.LPtrack(dt, track['trail_x'], track['trail_y'])

def solve_simdata(data, meta):
    sigma, r = meta.get('sigma'), meta.get('r')
    wavemodel = pwlpartition.sim_wavemodel(data, sigma)
    tsolver = pwltree.TreeSolver(data, overlap=True)
    tsolver.build_initial_tree(wavemodel)
    tsolver.build_priority()
    tsolver.solve(pwltree.stop_at(r))
    return tsolver

trackdata = mdl.LPtrack.from_track(track)
with support.Timer():
    tsolver = solve_simdata(trackdata, meta)

model = tsolver.get_model()

# %%

fig, ax = plt.subplots(figsize=(200,200))
pwlpartition.simple_model_plot(ax, model, data)
ax.plot(model.x, model.y, marker='d')
fig.savefig('sim_pwl.svg')


# %%
tsolver.history[-1], meta

# %%
