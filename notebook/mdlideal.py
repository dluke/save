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
# make a more ideal synthetic data  and see if it can be solved efficiently and accurately

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
notename = 'mdlideal'
mplstyle = {"font.size": 24}

# %% 
np.random.seed(0)

_l = 1.0
sigma = 0.15
length = synthetic.Uniform(_l, _l)
# angle = synthetic.Normal(scale=pi/4)
angle = synthetic.Normal(scale=pi/4)

dx = synthetic.Constant(_l/10)
error = synthetic.Normal(scale=sigma)

# test mirrored normal 
mnormal = synthetic.AlternatingMirrorNormal(loc=pi/4, scale=pi/16)

def exactplot(ax, gen, N=1000):
    basis = np.linspace(-pi, pi, N)
    p = list(map(gen.pdf, basis))
    ax.plot(basis, p)
    def_blue = '#1f77b4'
    ax.fill_between(basis, 0, p, alpha=0.2, hatch="/", edgecolor=def_blue)
    ax.set_xticks([-pi,0,pi])
    ax.set_xticklabels([r'$-\pi$','0',r'$\pi$'])

def genplot(ax, gen, N=10000):
    samp = gen.sample(N)
    sns.histplot(samp)
    ax.set_xlim(-pi/2, pi/2)

# genplot(plt.gca(), mnormal)
with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(6,4))
    # exactplot(ax, mnormal)
    exactplot(ax, angle)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$P$')
    # ax.set_xlabel(r'$\theta$')

# pub.save_figure("mdlideal_mirror_distribution", notename)

# %% 

N = 10
angle = mnormal
# pwl = synthetic.new_pwl_process(length, angle, N)
pwl = synthetic.new_static_process(length, angle, N)
synthdata = synthetic.sample_pwl(pwl, dx, error)

def save_data(data):
    target =  "annealing/current_data.pkl"
    print("writing to ", target)
    with open(target, 'wb') as f:
        pickle.dump(data, f)
save_data(synthdata)

# notebook specific plotting config
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


params = {"coarsen_sigma" : sigma, "angle_threshold" : 5}
wavemodel, lptr, meta = pwlpartition.initial_guess(synthdata.x, synthdata.y, params)
print('initial guess M', wavemodel.M)

# local_plot(wavemodel, synthdata, fs=20)
fs = 10
fig, ax = plt.subplots(figsize=(fs,fs))
pwlpartition.simple_model_plot(ax, wavemodel, synthdata, pwl)


def describe_pv(solver):
    np.set_printoptions(precision=5, suppress=True)
    pvalue = solver.partition.estimate_segment_probability()
    cdfv = solver.partition.get_segment_cdfv()
    print('pv', pvalue)
    print('cdf', cdfv)


# %% 
# * -------------------------------------------------------------------------------------
# * chi2 
# loss_conf = {'contour_term': 0.00, 'continuity_term': 0}
loss_conf = {"contour_term":0.01}

partition = pwlpartition.PartAnneal(wavemodel, synthdata, loss_conf=loss_conf, 
    use_alternative=True, use_probability_loss=True, use_bounds=False)
solver = pwlpartition.Solver(partition, r=meta['r'], sigma=meta['sigma'], 
    min_constraint=1, use_description_length=False)

describe_pv(solver)


fig, ax = plt.subplots(figsize=(20,20))
pwlpartition.simple_model_plot(ax, solver.partition.model, synthdata, pwl)
solver.dump_state("initial_state")

# %% 
# before any optimisation
solver.linear_solve()
solver.dump_state("initial_solve")

describe_pv(solver)

fig, ax = plt.subplots(figsize=(20,20))
pwlpartition.simple_model_plot(ax, solver.partition.model, synthdata, pwl)

# %% 
# * NEW FROM SAVE STATE
solver = pwlpartition.Solver.load_state("initial_solve")
solver.partition.loss_conf = {"contour_term":0.01}

describe_pv(solver)

fig, ax = plt.subplots(figsize=(10,10))
pwlpartition.simple_model_plot(ax, solver.partition.model, synthdata, pwl)

# %% 
control = {"p_threshold" : 0.98, "maxiter" : 500}
solver.chi2_solve(control)

# %% 
describe_pv(solver)

fig, ax = plt.subplots(figsize=(10,10))
pwlpartition.simple_model_plot(ax, solver.partition.model, synthdata, pwl)

# %% 
examine = False
if examine:
    _solver = solver.clone()
    _solver.partition.loss_conf = {}
    _solver.use_binary_percolate = True
    _solver.partition.delete_at(6)
    # _solver.anneal_around(6)
    _solver.partition.model.move_data_left_at(6, 1)
    _solver.local_optimise_at(6)
    # _solver.binary_percolate_at(6)
    describe_pv(_solver)

    fig, ax = plt.subplots(figsize=(10,10))
    pwlpartition.simple_model_plot(ax, _solver.partition.model, synthdata, pwl)


# %% 
control = {'maxiter': 500, 't_end': 0., 'tolerance': 1e-8, 'greedy':False}
# control['maxiter'] = 1
# output = pwlpartition.Output("mdlpartition/")
output = None

with support.Timer() as t:
    solver.priority_solve(control, output=output)


# %% 
# mdl_plot(solver, synthdata)

fig, ax = plt.subplots(figsize=(10,10))
pwlpartition.simple_model_plot(ax, solver.partition.model, data=synthdata, pwl=pwl)

# %% 
sigma, solver.sigma
solver._loss_history
pwlpartition.plot_solver_convergence(solver)
# %% 

solver.partition._cache_cdfv
pvalues = solver.partition.estimate_segment_probability()
pvalues
np.prod(pvalues)



