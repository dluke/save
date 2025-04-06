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
# We want to replace deviation.var with a suitable summary statistic for walking
# And our best idea is variance of aspect (aspect.var)
# so lets compute that for fanjin and for simulated data

# %% 
import warnings
import sys, os
import copy
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import pili
import readtrack
import command
import stats
import parameters
import rtw
import twutils

import pili.publication as pub

notename = 'walking_statistic'

# %% 

# twanalyse.out_of_plane(ltrs)

