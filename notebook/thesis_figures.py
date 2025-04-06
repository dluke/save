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
# generate various figures for papers or thesis

# %% 
# plot the surface potentials 

# %% 
import matplotlib.pyplot as plt
import matplotlib as mpl

import pywt

import sys
import os
join = lambda *x: os.path.abspath(os.path.join(*x))

import pili

# %% 
sys.path.append('/home/dan/usb_twitching/')
import thesis.publication as publication

# %% 
basestyle = {
    "font.size": 24,
    "text.usetex": True,
    "lines.linewidth" : 3
}

# %% 

[phi, psi, x] = pywt.Wavelet('db2').wavefun(level=4)

with mpl.rc_context(basestyle):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.axhline(0, color='k', alpha=0.2)
    l1, = ax.plot(x, psi)
    l2, = ax.plot(x, phi)
    
    ax.legend([l1,l2], ["$\psi(x)$", "$\phi(x)$"])


publication.save_figure("db2_wavelet")


# %% 
# * EXAMPLE TRAJECTORIES

import pwlstats


# load the candidate PWL model
path = join(pwlstats.root, "run/partition/candidate/_candidate_pwl/")
solver = pwlstats.load_solver_at(path)
partition = solver.partition
model = partition.model
data = partition.data
data.reset_dt()
short = data.cut_index(0,200)

# %% 

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.2, 'color': defcolor[2], 'markersize': 8}

with mpl.rc_context({"font.size" : 32}):
    fig, ax = plt.subplots(figsize=(20,20))
    ax.plot(data.x, data.y, label='data', **ptlkw)
    ax.set_aspect("equal")
    ax.set_xlabel('x ($\mu m$)')
    ax.set_ylabel('y ($\mu m$)')
    ax.set_axis_off()


def draw_ruler(ax):
    fig = ax.get_figure()
    fig.canvas.draw()
    length = 5.0
    rulerstyle = dict(linewidth=6.0, c='k', alpha=0.6)

    axis_to_data = ax.transLimits.inverted()
    px, py = axis_to_data.transform((0,0.2))
    px = data.x.min()

    ax.plot([px, px + length], [py, py], **rulerstyle)
    apx, apy = axis_to_data.transform((0,0.3))
    ax.annotate(r'$5.0 \mu m$', (px, apy), fontsize=32)

draw_ruler(ax)

# %% 

plot_target = join(pwlstats.root, "impress/images")
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.30, 'color': defcolor[2], 'markersize': 12}



fig, ax = plt.subplots(figsize=(12,12))
ax.plot(short.x, short.y, label='data', **ptlkw)
ax.set_aspect("equal")
ax.set_axis_off()

ax.set_xlabel("x")
ax.set_ylabel("y")



# %%
# plot first chunk for presentation purposes
data = partition.data
shortdata = partition.data.cut_index(0,200)
x, y = shortdata.get_n2().T


# pwlpartition.simple_model_plot(ax, model.cut(0,200), data= shortdata)

wavemodel, lptr, meta = pwlpartition.initial_guess(shortdata.x, shortdata.y)

defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1, c2, c3 = defcolors[:3] 
model_style = {"linestyle": '-', 'marker': 'D', 'lw':4, 'alpha':0.5, 'color':c2}
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.3, 'color':c3}

def local_simple_plot(ax, model, data):
    ax.plot(data.x, data.y, label='data', **ptlkw)
    ax.plot(model.x, model.y, label='wavelet', **model_style)
    ax.set_aspect('equal')
    # ax.legend(fontsize=20, loc=(1.04, 0))
    ax.legend(fontsize=20)


with mpl.rc_context({'font.size': 16}):
    fig, ax = plt.subplots(figsize=(12,5))
    local_simple_plot(ax, wavemodel, shortdata)
    ax.set_ylabel('y')
    ax.set_xlabel('x')

plot_target = join(pwlstats.root, "impress/images")
target = join(plot_target, "wavelet_candidate_example.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target)
