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
# construct a new local/global optimistion algorithm based using ideas from simulated annealing

# %% 
import os
from copy import copy, deepcopy
import random
import collections
join = os.path.join 
import numpy as np
pi = np.pi
norm = np.linalg.norm
import pickle
import logging

import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd

from tabulate import tabulate
import llist

import _fj

import shapeplot
import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

from pili import support
import mdl

import annealing 
from annealing import AnnealNode, AnnealSegmentList, Anneal, Solver


# %% 
# load data
synth_m = mdl.load_model("build_synthetic/synth_model")
synthdata = mdl.load_model("build_synthetic/synth_data")

# %%  [markdown]
# layout our algorithm
# ....

# %% 
# logging.basicConfig(filename=logfile, level=logging.DEBUG, filemode='w')

# %% 

# can come up with various strategies 
# the baseline will be to randomly choose segments
# and then randomly try either create/destroy
# and then after a local optimisation, we will accept the new state 
# if description length decreased 
# (optionally we will accept if description length is the same but M is reduced)
# if rejecting the state, we will reduce the "temperature" of the corresponding nodes 
# ! we should at least try create/destroy before reducing the temperature?
# ! implement faster local optimisation of the whole trajectory by optimising segments?

# how am I going to call this every time we run sections of the notebook?
from importlib import reload
reload(logging)

anneal = Anneal()

# # ! tmp 
# _synth_m = synth_m.cut(0,5)
# _synthdata = synthdata.cut(0,4.9)
# # ! ~tmp
_synth_m = synth_m
_synthdata = synthdata
anneal.initialise(_synth_m, _synthdata)

print(anneal.llmodel[0])


# %%  
# anneal.get_segment_node(7)
anneal.llmodel

# %%  
# test time mapping
def test_pwl_coord():
    scoord = anneal.seglist.get_local_coord()
    pwl, mapped = anneal.get_pwl_coord(get_mapped=True)

    fig, ax = plt.subplots(figsize=(10,10))
    _mapped = mapped
    _pwl = pwl
    mdl.plot_mapped(ax, _mapped, _synthdata, _synth_m)
# test_pwl_coord()


# %%  


# %%  
# rq_solver.create_and_destroy(32)
# def clear_logging():
#     for name in ["AnnealSegmentList", "Anneal", "Solver"]:
#         annealing.clear_log(name) 
# clear_logging()
# reload(logging)
# reload(annealing)

random.seed(0)
def test_random_queue_solve():
    rq_solver = Solver(anneal)
    rq_solver.random_queue_solve()
    _model = rq_solver.get_current_model()
    print(rq_solver._record)
    return rq_solver

rq_solver = test_random_queue_solve()


# %%  

def test_single_edit():
    sidx = 16
    rq_solver = Solver(anneal)
    rq_solver.create_and_destroy(sidx)
    _model = rq_solver.get_current_model()

# %%  

# ! hide in method
def modify(anneal):
    _midx = 6
    node = anneal.llmodel.nodeat(_midx)
    # nodes = [node]
    # node.value.x += 0.1
    # node.value.y += 0.1
    start_model = anneal.get_current_model().cut(0,10)

    # nodes = anneal.destroy(anneal.llmodel.nodeat(_midx))
    # nodes = anneal.destroy(anneal.llmodel.nodeat(_midx))
    # nodes = [anneal.llmodel.nodeat(_midx)]
    # print([n.value for n in nodes])

    nodes = anneal.create(anneal.llmodel.nodeat(_midx))

    # anneal.local_lsq_optimise(nodes, n_local=0)
    anneal.local_lsq_optimise(nodes, n_local=1)
    _model = anneal.get_current_model()

# %%  
_record = rq_solver._record
move = [_r[0] for _r in _record]
collections.Counter(move)

# %%  
inter = {}
# inter['is_outlier'] = anneal._outlier
_model = rq_solver.get_current_model()
inter['is_outlier'] = rq_solver.anneal._outlier
 
tmax = 200
fig, ax = plt.subplots(figsize=(20,20))
_m = _model.cut(0,tmax)
_d = _synthdata.cut(0,tmax)
mdl.plot_model_on_data(ax, _m, _d, intermediate=inter)
_synth = _synth_m.cut(0, tmax)
# ax.plot(start_model.x, start_model.y, c='gray', alpha=0.2, linewidth=10)
ax.plot(_synth.x, _synth.y, c='gray', alpha=0.2, linewidth=10)



# %%
