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
# Analyse FJ data for evidence of a "pivot" action.
# An action where one pilus fixes the leading pole and another causes a sharp rotation.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import _fj
import shapeplot
import plotutils
import twutils
import stats
import command
import readtrack
import twanalyse
import pilush

# %%
# example simulated data
notename = 'pivot'
notedir = os.getcwd()
fjdir = os.path.join(notedir, "~/usb_twitching/fanjin/working_copy")
pdir = notename+'/'
simdir = os.path.join(notedir, "exampledata/two_parameters/pilivar_0013.00000_k_spawn_05.00000/")
simdata = os.path.join(simdir, "data/")


# %%
debug = None
# idx, fjtrs = _fj.slicehelper.load_trs('all', debug)
idx, fjtrs = _fj.slicehelper.load_trs('high_displacement_crawling', debug)

# %%
# first we take a side step and analysis the body dimensions
dd = twanalyse.body_dimensions(fjtrs)

# %%
# plot mean distrubtions 
fig = plt.figure()
lax = fig.add_subplot(1, 2, 1)
lax.set_ylabel('P')
lax.set_xlabel('width')
# lax.hist(dd['mean_width'])
res = 100
plotutils.ax_kdeplot(lax, dd['mean_width'], res=res, hist=True)
print('mean width', np.mean(dd['mean_width']))
rax = fig.add_subplot(1, 2, 2)
rax.set_ylabel('P')
rax.set_xlabel('length')
plotutils.ax_kdeplot(rax, dd['mean_length'], res=res, hist=True)
# rax.hist(dd['mean_length'])
plt.show()


# %%
# plot gradients
fig = plt.figure()
lax = fig.add_subplot(1,2,1)
lax.set_ylabel('P')
lax.set_xlabel('width gradient')
plotutils.ax_kdeplot(lax, dd['w_gradient'])
plotutils.ax_kdeplot(lax, twutils.trim(dd['l_gradient'], 0.01))
lax.legend(lax.lines, ['width gradient', 'length gradient'])
plt.show()

# %% [markdown]
# The mean bacteria body width is 0.93 $\mu m$. As expected the gradient of width relative to time
# is firmly centered on 0. We will therefore take the mean width for the whole trajectory to be the
# unchanging width of the bacteria. Also expected is that the length gradients are positive which 
# is due to bacteria growth.

# %%
hdc_list = _fj.slicehelper.load('high_displacement_crawling')
cut_list = hdc_list
dd_pivot, dd_action = twanalyse.pivot_action(idx, fjtrs)
# %%
sd = twanalyse._pivot_summary_stats(fjtrs, dd_pivot, dd_action)
twutils.print_dict(sd)

# %%
hstyle = {'ec':'black', 'density':True}
def plot_dt_piv(dd, hstyle={}):
    trim_to = 20
    dt_arr = np.concatenate(dd['dt_all'])
    ax = plt.gca()
    dt_arr_clip = dt_arr[dt_arr< trim_to]
    ax.set_xlabel('pivot duration')
    h = ax.hist(dt_arr_clip, bins=dt_arr.size//10, **hstyle)
    return h
plot_dt_piv(dd_pivot, hstyle)
plt.show()


# %%
# plot step times
def plot_dt_att(dd, hstyle={}):
    dt_arr = np.concatenate(dd['dt_all'])
    ax = plt.gca()
    h = ax.hist(dt_arr, bins=np.arange(0, 5.0, 0.1), **hstyle)
    return h
plot_dt_att(dd_action, hstyle)
plt.show()

# %% 
def plot_dx_trail(ax, dd, hstyle={}):
    dx_trail_arr = np.concatenate(dd['dx_trail_all'])
    ax = plt.gca()
    h = ax.hist(dx_trail_arr, bins=20, **hstyle)
    ax.set_xlabel('trail distance')
    return h

plot_dx_trail(plt.gca(), dd_pivot, hstyle)
plt.show()


# %% [markdown]
# how sure are we that the leading and trailing poles are correctly identified?
# TODO
# %%
# since we don't have the original image data. It would be sensible to redraw the pivot actions
# and watch them as movies.
import twanimation
chose = 3053
tr = fjtrs[np.where(idx==chose)[0][0]]
def animate(tr):
    fig = plt.figure()
    sample = 10
    twanimation.outline(fig, [tr], sample=sample, camera='follow', savefile='fj_{:04d}.mp4'.format(chose))
#

# %% [markdown]
# We have defined the "pivot" the max distance travelled of the trailing pole while the leading pole is 
# stationary. Also within each pivot we identify the "action" which is a sharp turn.
# quantify the frequency and magnitude of pivot and the timescale associated with pivot actions
# 


# %%
sim_trs = readtrack.trackset(simdata)
sim_dd_pivot, sim_dd_action = twanalyse.pivot_action('all', sim_trs)

# %%
shstyle = {'ec':'black', 'density':True, 'color':'yellow'}
# %%
plot_dt_piv(sim_dd_pivot, hstyle=shstyle)
plt.title('Simulated')
plt.show()
# %%
plot_dt_att(sim_dd_action, hstyle=shstyle)
plt.title('Simulated')
plt.show()

# %%
ax = plt.gca()
shstyle['alpha'] = 0.5
shstyle['color'] = 'yellow'
h_pivot = plot_dx_trail(ax, sim_dd_pivot, hstyle=shstyle)
# shstyle['color'] = 'red'
# h_action = plot_dx_trail(ax, sim_dd_action, hstyle=shstyle)
# cannot use legend easily for histograms
plt.title('Simulated')
plt.show()
# %% [markdown]
# TODO revsit definiton of pivot and action
# we seem to have an issue where mean(dx_trail) for actions is > mean(dx_trail) for pivot
# this was not intended 

# %%
sd = twanalyse._pivot_summary_stats(sim_trs, sim_dd_pivot, sim_dd_action)
with command.chdir(simdir):
    print("writing stats ...")
    twutils.print_dict(sd)
    print("to local.json in directory {}".format(simdir))
    stats.extend(sd)

# %% [markdown]
# We still need to explicitly check the pili configurations that are responsible for pivot actions
# We can do this in simulation but of course not for experiments

# %%
with command.chdir(simdir):
    ptrs = readtrack.bound_piliset()

# %%
# call pilush._check_configuration
# collect output and draw distributions
# also compute the standard distributions for comparison
# import importlib; importlib.reload(pilush)
# dd_stat = pilush._check_configuration(ptrs, sim_trs, sim_dd_pivot, sim_dd_action)
dd_stat = pilush._check_configuration(ptrs, sim_trs, sim_dd_pivot, sim_dd_action)
# %%
import itertools
mean_npili = np.array(list(itertools.chain(*[dd['mean_npili'] for dd in dd_stat])))
track_max_theta = [dd['max_theta'] for dd in dd_stat]
track_min_theta = [dd['min_theta'] for dd in dd_stat]
max_theta = np.array(list(itertools.chain(*track_max_theta)))
min_theta = np.array(list(itertools.chain(*track_min_theta)))

# %% [markdown]
# we need reference distributions to compare to 
# For the angle reference distribution we need to compute min/max acos(ax_pilus.-e_z) for the whole track
# %% 
#
min_z_theta, max_z_theta = pilush._pili_reconstruction(ptrs, sim_trs)

# %% 
# can we skip a step here and used a fixed window size for generating reference distributions
import random
fixed_window = int(np.mean(np.concatenate(sim_dd_pivot['dt_all'])) / 0.1)
print('fixed_window', fixed_window)

nbound = [tr['nbound'] for tr in sim_trs]
N = int(1e5)
def sample_window(arr_list, N, _reduce, fixed_window=103, condition=None):
    n = len(arr_list)
    arr_size = [len(arr) for arr in arr_list]
    count = 0
    sample = np.zeros(N)
    while count < N:
        # choose random track
        track_i = random.randrange(0,n)
        # choose random position
        idx = random.randrange(arr_size[track_i]-fixed_window)
        chunk = arr_list[track_i][idx:idx+fixed_window]
        if condition and not condition(chunk):
            continue
        sample[count] = _reduce(chunk)
        count += 1
    return sample
sample_nbound = sample_window(nbound, N, np.mean, fixed_window=fixed_window)
def no_nan_condition(chunk):
    return not np.any(np.isnan(chunk))
sample_max_theta = sample_window(max_z_theta, N, np.nanmax, fixed_window, None)
sample_min_theta = sample_window(min_z_theta, N, np.nanmin, fixed_window, None)
# print(sample_max_theta)

# %% 
def plot_theta(ax, max_theta, min_theta, sample_max_theta, sample_min_theta):
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'P')
    res = 50
    lstyle = {'linewidth':4}
    h = False
    plotutils.ax_kdeplot(ax, max_theta, res=res, linekw=lstyle, hist=h)
    plotutils.ax_kdeplot(ax, min_theta, res=res, linekw=lstyle, hist=h)
    base_style = {'linestyle':'--'}
    lmin, lmax = ax.lines
    base_style['color'] = lmin.get_color()
    plotutils.ax_kdeplot(ax, sample_max_theta, res=res, linekw=base_style, hist=True)
    base_style['color'] = lmax.get_color()
    plotutils.ax_kdeplot(ax, sample_min_theta, res=res, linekw=base_style, hist=True)
    labels = [r'max($\theta$)', r'min($\theta$)']
    labels.append('reference '+labels[0])
    labels.append('reference '+labels[1])
    ax.legend(ax.lines,  labels)
plot_theta(plt.gca(), max_theta, min_theta, sample_max_theta, sample_min_theta)
plt.show()

# %% 

def plot_npili(ax, mean_npili, sample_nbound):
    ax.set_xlabel(r'$\langle N_{\mathrm{bound}} \rangle$')
    ax.set_ylabel(r'P')
    lstyle = {'linewidth':4}
    base_style = {}
    plotutils.ax_kdeplot(ax, mean_npili, res=50, linekw=lstyle, hist=True)
    plotutils.ax_kdeplot(ax, sample_nbound, res=100, linekw=base_style)
ax = plt.gca()
plot_npili(ax, mean_npili, sample_nbound)
ax.legend(ax.lines, ['pivot actions', 'reference sample'])
plt.show()

# %% [markdown]
# compute the pivot statistics for our two parameter case
