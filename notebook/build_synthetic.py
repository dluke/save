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
# construct synthetic twitching data for use in testing MDL-PiecewiseLinear modelling

# %% 
import os
join = os.path.join 
import numpy as np
pi = np.pi
norm = np.linalg.norm
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd

from tabulate import tabulate

import _fj

import shapeplot
import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

from pili import support
import mdl

from synthetic import *

# %% [markdown]
# need 3 parameters:
# * displacement length parameter / distribution
# * sampling time distribution
# * displacement angle parameter / distribution
# * measurement error function

# %% 

np.random.seed(0)

_l = 0.2
r = 0.03
length = Uniform(_l, _l)
angle = Normal(scale=pi/4)

dx = Constant(_l/4)
error = Normal(scale=r/2)

N = 10
pwl = new_pwl_process(length, angle, N)
synthdata = sample_pwl(pwl, dx, error)

# %%

conf = {}
_data = synthdata
par_value = _l
inter = mdl.new_intermediate()
coarsen_parameter = conf.get("coarsen_parameter", 'l')
_coarse = mdl.recursive_coarsen(_data, par_value, parameter=coarsen_parameter)
print("solving model with M = {}, N = {}".format(len(_coarse),len(_data)))

dl, model, inter = mdl.local_optimal_lsqmdl(synthdata, _l, r)

original = {'x': _data.x, 'y': _data.y}
# ax = mdl.plot_model_on_data(_coarse, original, intermediate=inter, config={"figsize":(20,5)})
mdl.new_plot_model_on_data(model, original, intermediate=inter, config={"figsize":(20,5)})


# %%
def new_plot_synthetic(pwl, synthdata):
    ax = plt.gca()

    fade_style = {"color":'k', "alpha":0.2}
    ax.plot(pwl.x, pwl.y, **fade_style)

    data_style = {"color":'b', "alpha":0.4, "linestyle":'--', "marker":'o'}
    ax.plot(synthdata.x, synthdata.y, **data_style)
    ax.set_aspect("equal")

def plot_synthetic(ax, pwl):
    fade_style = {"color":'k', "alpha":0.6, "linewidth":"2"}
    ax.plot(pwl.x, pwl.y, **fade_style)


fig, ax = plt.subplots(figsize=(20,5))
new_plot_synthetic(pwl, synthdata)


# %%

l0 = 0.2
bracket = mdl.new_Mbracket(_data, l0)
print("bracket", bracket)
Intf = mdl.mdlgss(_data, r, bracket)

# %%

m, dl = mdl.best(Intf)
print('best (m,dl) = ({},{})'.format(m, dl) )


# %%

sorted(Intf.cache_value.items())
# Intf.cache_model[]
_model = Intf.cache_model[10]
fig, ax = plt.subplots(figsize=(20,5))
plot_synthetic(ax, pwl)
mdl.plot_model_on_data(ax, _model, original, intermediate=inter)
pub.save_figure("example_synthetic_mdlmodel", "build_synthetic")


# %%
# ----------------------------------------------------------------
# scale up
N = 100
pwl = new_pwl_process(length, angle, N)
synthdata = sample_pwl(pwl, dx, error)
_data = synthdata


l0 = 0.2
bracket = mdl.new_Mbracket(_data, l0)
print("bracket", bracket)
Intf = mdl.mdlgss(_data, r, bracket)


# %%
def describe_convex_search(Intf):
    exect = sorted(Intf.cache_exect.items())
    total_exect = sum([ex[1] for ex in exect])
    print(f"total exec time {total_exect:.2f}s")
    

describe_convex_search(Intf)

# %%
m, dl = mdl.best(Intf)
_model = Intf.cache_model[m]
_original = {'x': _data.x, 'y': _data.y}
_inter = Intf.cache_inter[m]

mdl.save_model(_data, "build_synthetic/synth_data")
mdl.save_model(_model, "build_synthetic/synth_model")
mdl.save_model(_inter["coarse"], "build_synthetic/synth_initial_guess")

# %%
fig, ax = plt.subplots(figsize=(20,20))
mdl.plot_model_on_data(ax, _model, _original, intermediate=_inter)
plot_synthetic(ax,  pwl)
pub.save_figure("example_gssmdl_model", "build_synthetic", config={"svg":True})


# %%
print("M, True M", (len(_model), N))
Intf.cache_value

# %% [markdown] 
# IMPORTANT
# to recover this trajectory, use N = 100 and M = 55
# should take about 1 min to solve
# %%
# lets talk a quick look at the initial guess
with support.Timer() as t:
    _coarse = mdl.recursive_coarsen(_data, 55, parameter='M')
_coarse

fig, ax = plt.subplots(figsize=(20,20))
mdl.plot_model_on_data(ax, _coarse, _original, intermediate=_inter)
plot_synthetic(ax,  pwl)
pub.save_figure("example_gssmdl_initial_guess", "build_synthetic", config={"svg":True})


# %%
import scipy.stats
rv = scipy.stats.norm(scale=np.pi/4)
x = np.linspace(-np.pi,np.pi,1000)
plt.plot(x, rv.pdf(x))

