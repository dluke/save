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

# %%
# I don't really understand the Kolmogorov-Smirnov test.
# this is the place to play with that
import scipy.stats
import numpy as np

# %%
# first generate two identical distributions and check the result
N = int(1e6)
N_ = int(1e4)
data_1 = np.sin(np.random.rand(N) * np.pi)
data_2 = np.sin(np.random.rand(N_) * np.pi)
ks, pvalue = scipy.stats.ks_2samp(data_1, data_2, mode='exact')
print('ks statistic ', ks)
print('pvalue', pvalue)

