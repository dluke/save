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
# no tmos data, no analysis, just developing a way to split the axes from the graph content in mpl

# %% 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# %% 
xlim = (-np.pi, np.pi)
ylim = (-1, 1)
support = np.linspace(-np.pi, np.pi, 2001)
r1, rc = [1,32]
gridkw = {'width_ratios':[r1,rc], 'height_ratios':[rc,r1]}
fig, axes = plt.subplots(2,2, figsize=(5,5), gridspec_kw=gridkw)
_yax = axes[0,0]
_pax = axes[0,1]
_xax = axes[1,1]
_nonax = axes[1,0]
#
_pax.plot(support, np.sin(support))
_xax.set_xlim(xlim)
_yax.set_ylim(ylim)
def hide_ticks(ax):
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
def show_xonly(ax):
    for name in ['left', 'top', 'right']:
        ax.spines[name].set_visible(False) 
    ax.yaxis.set_ticks([])
def show_yonly(ax):
    for name in ['top', 'right', 'bottom']:
        ax.spines[name].set_visible(False) 
    ax.xaxis.set_ticks([])
_nonax.axis('off')
hide_ticks(_pax)
show_xonly(_xax)
show_yonly(_yax)
_xax.set_xlabel('x-axis')
_yax.set_ylabel('y-axis')
fig.tight_layout()