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
# see if we can make fast local optimisation of long trajectories 
# by optimising locally 

# more generally we should consider why scipy, i.e. Nelder Mead scales so badly 
# and what the stopping condition should  be

# %% 
import os
import random
import numpy as np
import pickle
join = os.path.join 
norm = np.linalg.norm

import scipy
import matplotlib.pyplot as plt

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

from pili import support
import mdl
import annealing 

# %% 
# 1. load the example synthetic data

# synth_m = mdl.load_model("build_synthetic/synth_model")
synthdata = mdl.load_model("build_synthetic/synth_data")
synthguess = mdl.load_model("build_synthetic/synth_initial_guess")
# %% 

mdl.new_plot_model_on_data(synthguess, synthdata)


# %% 

anneal = annealing.Anneal()

# # ! tmp 
# _synthmodel = synthguess.cut(0,5)
# _synthdata = synthdata.cut(0,5)
# # ! ~tmp
_synthmodel = synthguess
_synthdata = synthdata
anneal.initialise(_synthmodel, _synthdata)

outlier = anneal.get_outliers()
mdl.new_plot_model_on_data(_synthmodel, _synthdata, intermediate={"is_outlier":outlier})

# %% 
# the number we want to improve "per data point"
mres = np.mean(anneal.get_residuals())
# the number we want to improve locally but for the whole trajectory
res = np.sum(anneal.get_residuals())
# global optimisation 
dl = anneal.get_description_length()

# %% 
# come up with a procedure...
# e.g.

# locally optimise around each node sequentially and the do it a few times

anneal = annealing.Anneal()
anneal.initialise(_synthmodel, _synthdata)
before_res, before_dl = np.sum(anneal.get_residuals()), anneal.get_description_length()
solver =  annealing.ChunkLsq(anneal)

with support.Timer() as t:
    solver.linear_solve()

after_res, after_dl = np.sum(solver.anneal.get_residuals()), anneal.get_description_length()
print("LSQ residual")
print(before_res, after_res)
print("description length")
print(before_dl, after_dl)
outlier = solver.anneal.get_outliers()
mdl.new_plot_model_on_data(_synthmodel, _synthdata, intermediate={"is_outlier":outlier})

# %% [markdown]
# we compare this to the solution found by NM optimisation with default parameters
default_nm_model = mdl.load_model("build_synthetic/synth_model")
_anneal = annealing.Anneal()
_anneal.initialise(default_nm_model, synthdata)
_dl = _anneal.get_description_length()
print(_dl) # so the chunk optimisation did better 


# %% 
