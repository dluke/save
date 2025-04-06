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
# testing new implementation of pwl solver idea at pwlpartition.py

# %%

import annealing
import scipy.interpolate
from pwlpartition import wavelet_guess, percolate_cleanup
import pwlpartition
import pwlstats
import fjanalysis
import mdl
import _fj
from pili import support
import pili
import sctml.publication as pub
import sctml
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import os
import json
import numpy as np
import scipy.stats
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm


print("writing figures to", pub.writedir)


# %%
# trackidx = 1687
trackidx = 2924
track = _fj.trackload_original([trackidx])[0]
# fanjin wavelet track
fjtr = pwlstats.load_smoothed(trackidx)
pwlmodel = pwlstats.load_pwl_model(join(pwlstats.root, "run/_candidate_pwl"))

# %%
xy = track.get_head2d()
x, y = xy.T
# N = 100
N = 200
x = x[:N]
y = y[:N]
_data = {'x': x, 'y': y}
_lptr = mdl.LPtrack(None, x, y)

# sig = 0.04
sig = pwlpartition.estimate_error(x, y)
print(f'using sigma = {sig}')
denoised = pwlpartition.vary_sigma(x, y, sig)

angle_threshold = 15
angle_threshold = 30

wavemodel = pwlpartition.model_from_denoised(
    denoised, sigma=sig, angle_threshold=angle_threshold)

# %%


def local_plot(model, data=_lptr, fs=20):
    fig, ax = plt.subplots(figsize=(fs, fs))
    _config = {'match_data': True}
    pwlpartition.plot_wavemodel_on_data(ax, model, data, config=_config)

def mdl_plot(solver, data=_lptr, fs=20):
    partition = solver.partition
    is_outlier = solver.get_outliers()
    fig, ax = plt.subplots(figsize=(fs, fs))
    _config = {'h_outlier': True}
    mdl.plot_model_on_data(ax, partition.model, data, intermediate={
                           'is_outlier': is_outlier}, config=_config)


# %%
# fig, ax = plt.subplots(figsize=(fs, fs))
print('wavemodel has {} nodes'.format(len(wavemodel)))
local_plot(wavemodel)

# %%
# ----------------------------------------------------------------
# test priority solve

loss_conf = {'contour_term': 0.00, 'continuity_term': 0}

r = pwlpartition.estimate_r(sig)
partition = pwlpartition.PartAnneal(wavemodel, _lptr, loss_conf=loss_conf, 
    use_alternative=True, use_probability_loss=True, use_bounds=False, inter_term=True)
solver = pwlpartition.Solver(partition, r=r, sigma=sig, 
    min_constraint=1, use_description_length=False)

control = {'maxiter': 200, 't_end': 0., 'tolerance': 1e-8}
# control['maxiter'] = 1
output = pwlpartition.Output("mdlpartition/")

with support.Timer() as t:
    # solver.priority_solve(control, output=output)
    solver.chi2_solve(control, output=None)

# solver.dump_state("mdlpartition/shortchi2")

# %%

mdl_plot(solver)

# %%

solver.chi2_create_at(9, recurr=True)
# solver.chi2_create_at(10)

# %%
# pwlpartition.Solver.load_state("mdlpartition/confused")

# %%

# solver.partition.update_residuals()

# mdl_plot(solver)
local_plot(solver.partition.model)

solver.estimate_segment_probability()
print(solver.partition.get_segment_cdfv())
print(solver.partition.model.dt)



# %%
np.set_printoptions(precision=4, suppress=True)
solver.estimate_segment_probability()
print(solver.partition.get_segment_cdfv())

local_plot(solver.partition.model)

# %%
mdl_plot(solver)
# pub.save_figure("example_chi2_solve_candidate", "mdlpartition")

# %%
# ----------------------------------------------------------------
# test inter term

loss_conf = {'contour_term': 0.00, 'continuity_term': 0}

partition = pwlpartition.PartAnneal(wavemodel, _lptr, loss_conf=loss_conf, 
    use_alternative=True, inter_term=True)
r = pwlpartition.estimate_r(sig)
solver = pwlpartition.Solver(partition, r=r, sigma=sig, min_constraint=1)

control = {'maxiter': 300, 't_end': 0., 'tolerance': 1e-8, 'greedy':False}
# control['maxiter'] = 1
output = pwlpartition.Output("mdlanimate/", freq=10, allow_write=True)

with support.Timer() as t:
    solver.priority_solve(control, output=output)

mdl_plot(solver)

solver.save_state("mdlpartition/priority_solve")

# %%
defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.2, 'color': defcolor[2], 'markersize': 8}

partition = solver.partition
model = partition.model
fig, ax = plt.subplots(figsize=(12,4))
plt.plot(_lptr.x, _lptr.y, **ptlkw)
plt.plot(model.x, model.y)


# %%
def plot_solver_convergence(solver):
    score, loss = solver.get_history()
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    ax = axes[0]
    ax.plot(score)
    ax.set_ylabel('DL')
    ax = axes[1]
    ax.plot(loss.sum(axis=1))
    ax.set_ylabel('loss')

plot_solver_convergence(solver)

# %%
# ----------------------------------------------------------------

solver.partition.residual_at(1, 5)
solver.partition.inter_residual(1, 5)

# %%
solver.dump_state('tmp')

# %%

# %%
# ----------------------------------------------------------------
# test new annealing method # TODO move this to its own notebook

solver = pwlpartition.Solver.load_state('tmp')
solver.use_description_length = False
solver.partition._use_alternative = True
solver.partition.model.sidx = solver.partition.model._default_sidx()

# %%
control = {
    'p_threshold' : 0.99,
    'maxiter' : 1000 
}

with support.Timer():
    solver.chi2_solve(control)

mdl_plot(solver)

# %%
mdl_plot(solver)


# %%
# todo
# compute sample variance and compare to chi squared distribution
# for each segment in our solved model
residuals = solver.partition.get_residuals()
time = solver.partition.model.get_time()
n_list = []
sample = []

for i in range(time.size-1):
    ti, tf = time[i], time[i+1]
    n = tf - ti
    sample_residual = np.sum(residuals[ti:tf]**2/solver.sigma**2)
    chi = scipy.stats.chi2(n)
    p = 0.98
    alpha, beta = chi.ppf(1-p), chi.ppf(p)
    # contained = alpha < sample_residual and sample_residual < beta
    contained = sample_residual < beta
    cdfv = chi.cdf(sample_residual)
    print(n, cdfv, contained)

    n_list.append(n)
    sample.append(contained)

n_list = np.array(n_list)
sample = np.array(sample)

print(sample)


# %%

np.set_printoptions(precision=3, suppress=True)

def mdl_plot_highlight(solver, data=_data, fs=20):
    partition = solver.partition
    is_outlier = solver.get_outliers()
    fig, ax = plt.subplots(figsize=(fs, fs))
    pvalue = solver.estimate_segment_probability()
    print(pvalue)
    nodes = np.argwhere(pvalue < 0.02).ravel()
    print(nodes)
    _config = {'h_outlier': True, 'h_nodes': nodes}
    mdl.plot_model_on_data(ax, partition.model, data, intermediate={
                           'is_outlier': is_outlier}, config=_config)

mdl_plot_highlight(solver, _lptr, fs=20)
pub.save_figure("example_chi2_pre_solve", "mdlpartition")

# %%
_solver = solver.clone()
mdl_plot_highlight(_solver, _lptr, fs=20)


# %%
# try optimising chunks
si = 0; sf = solver.partition.model.M-1
_solver.chi2_optimise_chunk(si, sf, maxiter=200)
mdl_plot_highlight(_solver, _lptr, fs=20)

_solver.dump_state("tmp_chi2")
# %%
_solver = _solver.load_state("tmp_chi2")
mdl_plot_highlight(_solver, _lptr, fs=20)
pub.save_figure("example_chi2_pre_solve_optimised", "mdlpartition")

# %%
_solver.chi2_create_at(1)
mdl_plot_highlight(_solver, _lptr, fs=20)

# %%
_solver.chi2_create_at(2)
mdl_plot_highlight(_solver, _lptr, fs=20)

# %%
_solver.chi2_create_at(3)
mdl_plot_highlight(_solver, _lptr, fs=20)

# %%
_solver.chi2_create_at(4)
mdl_plot_highlight(_solver, _lptr, fs=20)

# %%
_solver.chi2_create_at(11)
mdl_plot_highlight(_solver, _lptr, fs=20)

# %%
next_index = np.argwhere(_solver.estimate_segment_probability() < 0.02).ravel()[0]
_solver.chi2_create_at(next_index)
mdl_plot_highlight(_solver, _lptr, fs=20)

# %%
_solver.dump_state("chi_nearly_optimal")

# %%
_solver = _solver.load_state("chi_nearly_optimal")
mdl_plot_highlight(_solver, _lptr, fs=20)
pub.save_figure("example_chi2_solve", "mdlpartition")


# %%
M = _solver.partition.model.M
_solver.chi2_optimise_chunk(M//2,M-1)

# %%
next_index = np.argwhere(_solver.estimate_segment_probability() < 0.02).ravel()[0]
_solver.chi2_create_at(next_index)
mdl_plot_highlight(_solver, _lptr, fs=20)


# %%
mdl_plot_highlight(_solver, _lptr, fs=20)
# _solver.estimate_segment_probability()





# %%
path = join(pwlstats.root, 'candidate_percolate_cleanup')
solver.dump_state(path)

mdl_plot(solver, data, fs=200)

# %%

# %%
# ----------------------------------------------------------------
#  we can use this idea to standardise our choice of r threshold across trajectories
#  with different spatial error
# stats normal distribution
sigma = sig
print('sigma', sigma)

normal = scipy.stats.norm(scale=sigma)
basis = np.linspace(-3*sigma, 3*sigma, 1000)
plt.plot(basis, normal.cdf(basis))


def p_contained(r):
    # probability of single normal distributed sample being contained in [-r, r]
    return 1 - 2 * normal.cdf(-r)
    # todo invert this function

print(p_contained(sigma))
print(p_contained(0.031))

basis = np.linspace(0, 3*sigma, 1000)
plt.plot(basis, [p_contained(x) for x in basis])

# %%
# invert numerically
rbase = np.linspace(0, 5*sigma, 1000)
evf = np.array([p_contained(r) for r in rbase])
plt.plot(evf, rbase)

f = scipy.interpolate.interp1d(evf, rbase)
# choose a probability threshold  i.e.
p = 0.98
r = float(f(p))
print('r = ', r)

