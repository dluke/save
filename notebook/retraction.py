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
# This notebook was created to answer a single question
# what if dwell_time is small? Can we fit fanjin crawling distribution like
# we can with anchor_angle_smoothing_fraction parameter?
# We can use sobol or mc search to answer (we will use sobol)

# %% [markdown]
# The first question is how small should we set dwell_time?
# lets load our mc4d data and recover the contract_length.mean 
# summary statistic. Then set min dwell_time to be c*contract_length/retraction_speed
# where c is some factor < 1.0.

# %% 
import os, sys
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import pandas as pd
import stats
import json
import matplotlib.pyplot as plt
#
import pili
import _fj
import fjanalysis
import sobol
import twanalyse
import twutils

# %% 
notedir, notename = os.path.split(os.getcwd())
root = pili.root
# candidate to compare against
simdir = join(root, "../run/5bfc8b9/cluster/mc4d")

# %% 
# loading simulated data
lookup = sobol.read_lookup(simdir)
problem = sobol.read_problem(simdir)
print(problem)
_ , lduid = sobol.collect([], targetdir=simdir, alldata=True)

# %% 
# load fanjin
all_idx, ltrs = _fj.slicehelper.load_linearized_trs("all")
reference_idx = _fj.load_subset_idx()
subsets = list(reference_idx.keys())

# %% 
# for each subset get the best fit simulated trajectory (ks_statistic) 
ks_scores = ['fanjin.%s.ks_statistic' % subset for subset in subsets]
Yf = sobol.collect_obs(lookup, lduid, subsets, ks_scores)
sortdf = sobol.sortscore(problem, lookup, Yf, ks_scores)
sortdf["top"]

# %% 
# pull contract_length.mean from simulated summary data
for subset in subsets:
    best = sortdf[subset].iloc[0]
    ld = lduid[best["dir"]]
    print(best["dir"])
    getter = twutils.make_get("contract_length.mean")
    clmean = getter(lduid[best["dir"]])
    print(subset, clmean)

# %% 
# looking at 
# ~/usb_twitching/run/5bfc8b9/cluster/mc4d/_u_WzGvK2wO
# taut_delay  = 0.03
# 0.03 + 0.05/0.75 \approx 0.10
# then set dwell_time min to 0.05
