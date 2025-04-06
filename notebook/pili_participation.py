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
# A quick notebook to analyse nbound and bound pili participation for k_ext_off and pilivar

# %% 
import os, sys
import numpy as np
import matplotlib.pyplot as plt
#
import txtdata
import rtw
join = os.path.join

# %% 
# paths
notedir = os.path.dirname(__file__)
rundir = os.path.abspath(join(notedir, '../../run'))
simdir = join(rundir, "b2392cf/k_ext_off")

# %% 
# load data
dc = rtw.DataCube(target=simdir)
ldata = dc.load_local()

# %% 
# bound pili participation
fig, axes = plt.subplots(2,1,figsize=(8,12))
ax1, ax2 = axes
vstyle = {"linestyle":'--', "color":'k'}
def _plot_bpp(ax):
    bpp = [ld['bound_pili_participation'] for ld in ldata]
    basis = 1.0/np.array(dc.basis[0])
    ax.plot(basis, bpp, marker='D')
    ax.set_ylabel('bound pili participation')
    ax.set_xlabel(txtdata.longnames.get(dc.pnames[0]))
    ax.axvline(1.0/0.625, **vstyle)
_plot_bpp(ax1)

# nbound
def _plot_nbound(ax):
    nbound = [ld['nbound']['mean'] for ld in ldata]
    std_err = np.array([ld['nbound']['std_error'] for ld in ldata])
    ax.set_xlabel(txtdata.longnames.get(dc.pnames[0]))
    style = {"elinewidth":2}
    ax.errorbar(basis, nbound, 1.96*std_err, **style)
    ax.set_ylabel('nbound')
    ax.axvline(1.0/0.625, **vstyle)
_plot_nbound(ax2)

plt.tight_layout()
plt.show()

# %% [markdown]
# We see that for these parameters that in the analysis range 
# the nbound and bound pili participation fraction change by
# a little more than 2x. The two values are heavily correlated as expected

# we could continue and check participation fraction for pilivar
# but it would be better to do on saltelli dataset
