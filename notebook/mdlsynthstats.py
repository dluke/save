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
# counter-part to mdlstatistics.py for synthetic data

# %% 
import os
import json
import pickle
import numpy as np
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd
from glob import glob

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili
import mdl
import annealing 

import pwlstats
import pwlpartition

# %% 
notename = "mdlsynthstats"
publish = False

# %% 
# operate on data object
def get_array(getter, data):
    return np.array([getter(local) for local in data])

def get_list(getter, data):
    return [getter(local) for local in data]

def make_get(localvar):
    keys = localvar.split('.')
    def getter(localdata):
        _dct = localdata
        for k in keys:
            _dct = _dct.get(k, None)
            if _dct is None: 
                return np.nan
        return _dct
    return getter


# %% 
# variant_target = join(pwlstats.root, "run/synthetic/vary_sigma")
# variant_target = join(pwlstats.root, "run/synthetic/chi2/vary_sigma")
# variant_target = join(pwlstats.root, "run/synthetic/inter_vary_sigma")
variant_target = join(pwlstats.root, "run/synthetic/vary_angle")
variant_target

# %% 

def load_json(at):
    with open(at, 'r') as f: return json.load(f)

def load_pkl(at):
    with open(at, 'rb') as f: return pickle.load(f)


def load_at(variant_target):
    config  = load_json(join(variant_target, "config.json"))
    data_dir = config["data_directories"]
    def _process(directory):
        dct = {}
        target_dir = join(variant_target, directory, "solver") 
        solver = pwlpartition.Solver.load_state(target_dir)

        # !tmp
        solver.partition.use_probability_loss = False 
        solver.partition.inter_term = False 
        # !~tmp

        dct['solver'] = solver
        initial_dir = join(variant_target, directory, "initial_guess") 
        dct['initial_model'] = pwlpartition.Solver.load_state(initial_dir).partition.model
        dct['model'] = dct["solver"].partition.model
        dct['truth'] = load_pkl(join(variant_target, directory, "truth.pkl"))
        dct['local'] = pwlstats.solver_summary(dct["solver"])
        dct['local_config'] = load_json(join(variant_target, directory, "config.json"))
        return dct
    datalist = [_process(directory) for directory in data_dir]

    # post process
    sampling_dx = get_array(make_get('local_config.params.sigma'), datalist).min()
    print("compute mean separation with sampling dx =", sampling_dx)
    for dct in datalist:
        meta = dct["local_config"]["meta"]
        meta["sampling_dx"] = sampling_dx
        dct['mdiff'] = pwlstats.diff_models(dct["model"], dct["truth"], meta)
        dct['initial_diff'] = pwlstats.diff_models(dct["initial_model"], dct["truth"], meta)

    return config, datalist

config, datalist = load_at(variant_target)

truedata = [pwlstats.true_summary(dct['truth']) for dct in datalist]


# %% 
# plot model accuracy
mplstyle = {"font.size": 20}
mpl.rcParams.update(mplstyle)


# %% 
# well , what is our estimate  of r/l for candidate?

analyse_candidate = False
if analyse_candidate:
    target = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
    def load_summary_at_target(target):
        solver = pwlpartition.Solver.load_state(join(target, "solver"))
        # !tmp !patch the old object
        solver.partition.inter_term = False 
        return pwlstats.solver_summary(solver)

    candidate_summary = load_summary_at_target(target)
    candidate_sigma = candidate_summary["estimate_sigma"]
    candidate_r = pwlpartition.estimate_r(candidate_sigma)
    draw_sigma_l = candidate_sigma / candidate_summary["mean_step_length"]
    draw_rl = candidate_r / candidate_summary["mean_step_length"]
    # todo use 0.05 quantile instead of mean ~ we want to find all the segments
    print("esimate experimental sigma/l {:.4f}/{:.4f} = {}".format(candidate_sigma , candidate_summary["mean_step_length"], draw_sigma_l))
    print("esimate experimental r/l", draw_rl)


# %% 
# did we sucessfully estimate the spatial error of synthetic data?

base_sigma = get_array(make_get("local_config.params.sigma"), datalist)
est_sigma = get_array(make_get("local_config.meta.sigma"), datalist)
print('true and estimated spatial error')
print('base', base_sigma)
print('estimate', est_sigma)
# slighly overestimated, but pretty good
print('r_threshold')
r = np.array([dct["local_config"]["meta"]["r"] for dct in datalist])
print(r)

with mpl.rc_context(mplstyle):
    fig, ax = plt.subplots(figsize=(8,4))
    y = est_sigma - base_sigma
    y_perc = y/base_sigma 
    ax.plot(config['basis'], y_perc, marker='D')
    xlabel = config['variant']
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$(\sigma^\prime - \sigma)/\sigma$')


if publish:
    pub.save_figure("ideal_sigma_estimate", notename)


# %% 

print('N = ', get_array(lambda dct: dct["truth"].M, datalist))

def plot_segment_accuracy(ax, config, datalist):
    accuracy = 100 * get_array(lambda dct: dct["model"].M, datalist) / get_array(lambda dct: dct["truth"].M, datalist)
    initial_accuracy = 100 * get_array(lambda dct: dct["initial_model"].M, datalist) / get_array(lambda dct: dct["truth"].M, datalist)

    r = np.array([dct["local_config"]["meta"]["r"] for dct in datalist])
    sigma = np.array([dct["local_config"]["meta"]["sigma"] for dct in datalist])
    mean_step = get_array(lambda dct: dct["truth"].get_step_length().mean(), datalist)

    xlabel = r"$\sigma/\langle l \rangle$"
    if analyse_candidate:
        ax.plot(draw_sigma_l, 10, marker='o', color='black', markersize=6)
        ax.plot([draw_sigma_l, draw_sigma_l], [0, 10], linestyle='--', color='black', lw=1)
        ax.annotate(rf"candidate {xlabel}", (draw_sigma_l, 10), fontsize=16)

    ax.axhline(100, c='black', linestyle='--', alpha=0.6)
    ax.plot(sigma/mean_step, initial_accuracy, marker='D', label='initial')
    ax.plot(sigma/mean_step, accuracy, marker='D', label='solve')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count segments (%)")
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.legend()

def plot_separation_accuracy(ax, config, datalist):
    accuracy = get_array(make_get('mdiff.mean_separation'), datalist)
    initial_accuracy = get_array(make_get('initial_diff.mean_separation'), datalist)

    r = np.array([dct["local_config"]["meta"]["r"] for dct in datalist])
    sigma = np.array([dct["local_config"]["meta"]["sigma"] for dct in datalist])
    mean_step = get_array(lambda dct: dct["truth"].get_step_length().mean(), datalist)

    ax.plot(sigma/mean_step, initial_accuracy, marker='D', label='initial')
    ax.plot(sigma/mean_step, accuracy, marker='D', label='solve')
    ax.set_xlabel(r"$\sigma/\langle l \rangle$")
    ax.set_ylabel("mean separation")
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(10,4))
ax = axes[0]
plot_segment_accuracy(ax, config, datalist)
ax = axes[1]
plot_separation_accuracy(ax, config, datalist)
fig.tight_layout()


if publish:
    # fname = '_'.join([config['variant'], 'accuracy'])
    fname = 'sigma_accuracy'
    pub.save_figure(fname, notename)


# %%

def plot_segment_accuracy(ax, config, datalist):
    accuracy = 100 * get_array(lambda dct: dct["model"].M, datalist) / get_array(lambda dct: dct["truth"].M, datalist)
    initial_accuracy = 100 * get_array(lambda dct: dct["initial_model"].M, datalist) / get_array(lambda dct: dct["truth"].M, datalist)

    x = config['basis']

    xlabel = r"$ std(\theta) $"

    ax.axhline(100, c='black', linestyle='--', alpha=0.6)
    ax.plot(x, initial_accuracy, marker='D', label='initial')
    ax.plot(x, accuracy, marker='D', label='solve')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count segments (%)")
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.legend()

def plot_separation_accuracy(ax, config, datalist):
    accuracy = get_array(make_get('mdiff.mean_separation'), datalist)
    initial_accuracy = get_array(make_get('initial_diff.mean_separation'), datalist)

    x = config['basis']
    xlabel = r"$ std(\theta) $"

    ax.plot(x, initial_accuracy, marker='D', label='initial')
    ax.plot(x, accuracy, marker='D', label='solve')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean separation")
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(10,4))
ax = axes[0]
plot_segment_accuracy(ax, config, datalist)
ax = axes[1]
plot_separation_accuracy(ax, config, datalist)
fig.tight_layout()


publish = True
if publish:
    fname = 'angle_sigma_accuracy'
    pub.save_figure(fname, notename)


# %%
# * compare the empirical and true angle distributions

empirical_angle_distrib = get_list(make_get('local.angles'), datalist)
true_angle_distrib = get_list(make_get('angles'), truedata)
empangle= np.abs(np.concatenate(empirical_angle_distrib))
trueangle = np.abs(np.concatenate(true_angle_distrib))

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(6,4))
hstyle = {'alpha':0.4}
sns.histplot(empangle, color=defcolor[0], label='measured', ax=ax, **hstyle)
sns.histplot(trueangle, color=defcolor[1], label='truth', ax=ax, **hstyle)
ax.legend()
ax.set_xlabel(r'$\theta$')
ax.set_xticks([0, np.pi/2, np.pi])
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
print(empangle.size, trueangle.size)
min_angle = empangle.min() * 180/np.pi
print('minimum measured angle', min_angle)

# todo: here are the distributions but what is the resolution?
# todo: we can compare breakpoint by breakpoint to see which ones missed
