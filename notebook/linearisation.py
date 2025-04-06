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
# Linearisation procedure as described by [Fanjin et al.](https://www.pnas.org/content/108/31/12617).

# %%
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import _fj
import plotutils
import twutils
import command
import shapeplot

# %%
# example simulated data
notename = 'linearisation'
notedir = os.getcwd()
simdir = os.path.join(notedir, "exampledata/two_parameters/pilivar_0013.00000_k_spawn_05.00000/")
simdata = os.path.join(simdir, "data/")
fjdir = os.path.join(notedir, "~/usb_twitching/fanjin/working_copy")
pdir = notename+'/'

# %%
# plot linearised data over original 
data_id = 1279
track = _fj.trackload([data_id])[0]
# ltrack = _fj.lintrackload([data_id])[0]
ltrack = _fj.linearize(track)
ax = plt.gca()
kw = {'alpha':0.5}
shapeplot.longtracks(ax, [track], linekw=kw)
shapeplot.longtracks(ax, [ltrack], linekw=kw)
savesvg = os.path.join(pdir, 'super_{:04d}.svg'.format(data_id))
print()
print('saving to ', savesvg)
plt.savefig(savesvg, format='svg')
# %%
# compute velocity and step velocity
track_velocity = track.get_head_v()
track_speed = np.linalg.norm(track_velocity, axis=1)
print('track_speed', np.mean(track_speed))

step_idx = np.array(ltrack.step_idx)
step_velocity = ltrack.get_step_velocity()
step_speed = np.linalg.norm(step_velocity, axis=1)
# weighted mean for all parameters on the stepped data !
# print('step speed', np.sum(dt * step_speed)/np.sum(dt) )


# %%
# load fanjin 
debug = None
debug = 100
# idx, fjtrs = _fj.slicehelper.load_trs('default_crawling_list', debug)
idx, fltrs = _fj.slicehelper.load_linearized_trs('default_crawling_list', debug)

# %%
print('{} tracks'.format(idx.size))
step_n_s = [np.diff(ltr.step_idx) for ltr in fltrs]
# will be nan if N segments is 0 
step_n_mean_s = np.array([np.mean(step_n) for step_n in step_n_s])
N_s = np.array([l.size for l in step_n_s])
N = np.mean(N_s)
step_n_mean = np.mean(step_n_mean_s)
# %%
# distribution of mean number of segments
ax = plt.gca()
vstyle = {'linestyle':'--', 'linewidth':2, 'color':'k'}
def plot_N_s(ax):
    ax.hist(np.array(N_s), bins=20, density=True)
    print('plotting N_s')
    print('mean ', N)
    print('lims ', np.min(N_s), np.max(N_s))
    data_threshold = 20
    low_idx = idx[N_s < data_threshold]
    print('with less than {} segments {}/{}'.format(data_threshold, low_idx.size, idx.size))
    ax.set_ylabel('P')
    ax.set_xlabel(r'N linearized segments (per track)')
    ax.axvline(N, label='mean', **vstyle)
    command.saveplt('N_s', pdir)
plot_N_s(ax)
plt.show()
# %%
# distribution of segment-timesreload(plotutils)
ax = plt.gca()
import matdef
s_time_m = matdef.TIMESTEP * step_n_mean_s
s_time_m_cut = twutils.trim_tail(s_time_m, 0.01)
def plot_mean_segment_tau(ax):
    print('plotting mean segment-time distribution')
    print('mean', np.nanmean(s_time_m_cut))
    print('lims ', np.nanmin(s_time_m_cut), np.nanmax(s_time_m))
    plotutils.ax_kdeplot(ax, s_time_m_cut, res=40, hist=True)
    ax.set_xlabel(r'mean segment $\tau$ per track (s)')
    ax.set_ylabel('P')
    command.saveplt('mean_segment_tau', pdir)
plot_mean_segment_tau(ax)
plt.show()

# %%
# pull out some 'random' tracks that have good data 
ax = plt.gca()
def plot_individual_tau(ax):
    data_threshold = 100
    low = np.nanquantile(s_time_m, 0.25)
    high = np.nanquantile(s_time_m, 0.75)
    tau_condition = np.logical_and(s_time_m > low, s_time_m < high)
    good_snm_idx = np.nonzero(np.logical_and(N_s > data_threshold, tau_condition))[0]
    request_n = 10
    res = 100
    for i in good_snm_idx[:request_n]:
        stat = matdef.TIMESTEP * step_n_s[i]
        plotutils.ax_kdeplot(ax, stat, res=res)
    ax.set_xlabel(r'$\Delta t$ of segments')
    ax.set_ylabel(r'P')
plot_individual_tau(ax)
plt.show()


# %%
# Run the same for simulation data
import readtrack
trs = readtrack.trackset(ddir=simdata)
ltrs = [_fj.linearize(tr) for tr in trs]
print('{} simulated tracks'.format(len(ltrs)))

# %%
step_n_s = [np.diff(ltr.step_idx) for ltr in ltrs]
N_s = [l.size for l in step_n_s]
N = np.mean(N_s)
def _compute_s_time(ltr):
    t_cut = ltr['time'][ltr.step_idx]
    return t_cut[1:] - t_cut[:-1]
s_time = [_compute_s_time(ltr) for ltr in ltrs]
s_time_all = np.concatenate(s_time)


# %%
# plot simulated s_time distribution
ax = plt.gca()
# s_time_cut = s_time_all
s_time_cut = twutils.trim_tail(s_time_all, 0.02)
res = 100
plotutils.ax_kdeplot(ax, s_time_cut, res=100, hist=True)
s_time_median = np.median(s_time_all)
#
ax.axvline(s_time_median, **vstyle)
ax.set_xlabel(r'$\Delta t$ of segments')
ax.set_ylabel(r'P')
print('median ', s_time_median)
command.saveplt('simulated_segment_time', pdir)
plt.show()
# %% [markdown]
# For this simulated data, segment times are sharply distributed.
# It might be useful to check $\langle N_\mathrm{ntaut}\rangle$ and for what 
# proportion of our trajectory do we have $N_\mathrm{ntaut} = 0$
# %%
def _have_taut_prop(tr):
    return 1.0 - np.count_nonzero(tr['ntaut'] == 0)/tr.size
taut_prop = np.mean([_have_taut_prop(tr) for tr in ltrs])
print('taut_prop', taut_prop)
print('< ntaut >', np.mean([np.mean(tr['ntaut']) for tr in ltrs]))

# %% [markdown]
# We see a large spike at $\Delta t = 0.2$ which is to be expected since the segementation distance is 0.12 $\mu m$
# and the retraction velocity is 0.75 $\mu m s^{-1}$ so bacteria can typically displace by 0.15 $\mu m$ in 0.2s.
# The question then becomes why this spike is *not* seen in FJ data (and why FJ bacteria are slower overall).
# Two possible options come to mind:  
# - Bacteria are usually prevented from displacing by more than 0.12 $\mu m$ in a single pili retraction by the
# concerted action of multiple bound pili
# - Bacteria generally displace by less than 0.12 $\mu m $ in a single action, implying that the typical taut lifetime
# of a pilus is less than 0.12/0.75 = 0.16 seconds (or that its binding distance is extremely small)

# %%
# 
import eventanalyse
with command.chdir(simdir):
    sdata = eventanalyse.lifetime()

# %%
print('simulation mean bound lifetime {}'.format( sdata['bound_time']['mean']))
# print('simulation mean taut lifetime {}'.format( sdata['taut_lifetime']['mean']))
print('simulation taut pili extension ratio {}'.format( sdata['taut_pili_extension_ratio']))
print('simulation mean taut contraction {}'.format( sdata['contract_length']['mean']))
# %% [markdown]
# mean bound lifetime $\tau_{\mathrm{bound}} = 2.28s$ the configured $\tau_{\mathrm{dwell}} = 1s$ indicating
# that the simulation suffers in some way from the detachment rules. Analysis of detachment rules needs its own notebook.  
# The mean taut lifetime $\tau_{\mathrm{taut}}$ is 1.4s but the contraction mean distance is only 
# $\Delta l = 0.15 \mu m$. Is this because a jammed state is reached? Is this jammed state caused by
# multiple pili or one pilus in the vertical configuration.

