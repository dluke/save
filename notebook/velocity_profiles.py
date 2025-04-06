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
# analysis and plotting of velocity profiles for simulated and fanjin data

# we examine velocity profiles for simulated and experimental data
# and for two dimensions only (k_spawn, pilivar) 
# we construct pareto optimal parameters and a metric using 
# (lvel.mean, lvel.std)
# After sobol analysis we have moved beyond this type of analysis

# this notebook outputs individual trajectories 
# and velocity distributions to ./velocity_profiles
# for the purpose of choosing the whitelist

# %%
verbose = False
work = False

# %%
from tqdm import tqdm
import os
join = os.path.join
import json
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib as mpl
import _fj
import plotutils
import command
import twutils
import twanalyse
import stats
import readtrack
import parameters
import scipy
import shapeplot
import pili




# %%
# paths
notename = "velocity_profiles"
pdir = notename+'/'
notedir = join(pili.root, 'notebook/')
rsimdir = "exampledata/two_parameters/pilivar_0013.00000_k_spawn_00.50000/"
# rsimdir = "exampledata/two_parameters/pilivar_0013.00000_k_spawn_05.00000/"
simdir = os.path.join(notedir, rsimdir)

# %%
# fanjin
debug = 100
debug = None
if debug is not None:
    print('running in debug mode...')
all_idx, ltrs = _fj.slicehelper.load_linearized_trs('all')
idx = _fj.slicehelper.load("default_crawling_list")
fltrs = [ltrs[i] for i in idx]

# %%
# plotting fj data
fjtrack_vel = [twanalyse._inst_vel(tr) for tr in fltrs]
fjtrack_mean = np.array([np.mean(v) for v in fjtrack_vel])
allvel = np.concatenate(fjtrack_vel)

# %%
# The same for simulation data
simdata = os.path.join(simdir, "data/")
trs = readtrack.trackset(ddir=simdata)

for tr in trs:
    tr._clean_bad_dt()
ltrs = [_fj.linearize(tr) for tr in trs]

track_vel = [twanalyse._inst_vel(tr) for tr in ltrs]
simvel = np.concatenate(track_vel)

# %% [markdown]
# Comparing instantaneous velocity of linearized tracks.
# Tracks linearised with respect to a threshold distance 0.12 microns. Instantaneous velocity
# is calculated over 0.1 second interval.

# %%
fjvstat = stats.stats(allvel)
simvstat = stats.stats(simvel)
print("Instantaneous velocity")
print(r"Fanjin")
twutils.print_dict(fjvstat)
print(r"Simulated example [{}]".format(rsimdir))
twutils.print_dict(simvstat)

# %%
import rtw
import txtdata

target = join(notedir, "../../run/two_parameter_model/two_parameters/")
dc = rtw.DataCube(target)
lvel_mean = dc.get_local_array( rtw.make_get("lvel.mean") )
lvel_std = dc.get_local_array( rtw.make_get("lvel.std"))
fjvmean = fjvstat['mean']
fjvstd = fjvstat['std']
rel_lvel_mean = lvel_mean - fjvmean
rel_lvel_std = lvel_std - fjvstd
print("Want to analyse a 2d parameter search dataset with parameters")
# table = [ [name] + base  for name, base in zip(dc.pnames, dc.basis) ]
print(dc.pnames[0], dc.basis[0])
print(dc.pnames[1], dc.basis[1])

# %%

plt.style.use(plotutils.get_style('image'))
def relative_image(expval, localdataname):
    ax = plt.gca()
    meanget = rtw.make_get(localdataname)
    def rmeanget(ld):
        return abs(meanget(ld) - fjvmean)
    def rmeanget_an(ld):
        return meanget(ld) - fjvmean

    rtw._param_image(ax, dc, rmeanget, annotate=True, 
        annotate_form=rtw.anform[localdataname], use_lognorm=True, _getter_an=rmeanget_an)
print("mean velocity relative to FJ data")
relative_image(fjvmean, 'lvel.mean')
plt.show()

print("velocity std relative to FJ data")
relative_image(fjvstd, 'lvel.std')
plt.show()
# %%
# so ok there are tracks with similar velocity and standard deviation but we have seen their shapes
# are not so close as we might like.
# we need to quantify this. We can use smoothing kernel to probability distribution and then
# do some kind of weighted least squares but I would like a better approach.
# go to higher order than 2nd momement? Just look at quantiles? KS statistic? 
pass


# %% [markdown]
# We want consider multiple metrics, in this case just mean velocity and standard deviation 
# initially. Lets start by finding the pareto set.
# %%
# pareto
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
from twutils import is_pareto_efficient_simple

# being above or below the experimental data is bad so take absolute value

objective_shape = rel_lvel_mean.shape
mean_objective = np.abs(rel_lvel_mean)
std_objective = np.abs(rel_lvel_std)


objectives = np.column_stack( [ mean_objective.flatten(), std_objective.flatten() ] )
pareto_front = is_pareto_efficient_simple(objectives)
pareto_front = pareto_front.reshape(objective_shape)
pareto_idx = np.nonzero(pareto_front)

# draw pareto front
im_front = np.full(objective_shape, 1.0)
im_front[pareto_idx] = 0.5

ax = plt.gca()

ax.set_xticks(np.arange(len(dc.slice_basis[1])))
ax.set_yticks(np.arange(len(dc.slice_basis[0])))
ax.set_xticklabels(dc.slice_basis[1])
ax.set_yticklabels(dc.slice_basis[0])
ax.set_xlabel(txtdata.prettynames.get(dc.pnames[1]))
ax.set_ylabel(txtdata.prettynames.get(dc.pnames[0]))

norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
Image = ax.imshow(im_front, norm=norm, cmap=plt.cm.gray, origin='lower')
plt.show()

# what is the difference between min and max in pareto front
def minmax(arr):
    return np.min(arr), np.max(arr)
min_mean, max_mean = minmax(rel_lvel_mean[pareto_idx])
min_std, max_std = minmax(rel_lvel_std[pareto_idx])
amin_mean, amax_mean = minmax(mean_objective[pareto_idx])
amin_std, amax_std = minmax(std_objective[pareto_idx])
# 
print('pareto front lims')
headers = ['', 'min', 'max', '', 'min', 'max']
table = tabulate([
    ['lvel mean ', min_mean, max_mean, 'absolute', amin_mean, amax_mean],
    ['lvel abs ', min_std, max_std, 'absolute', amin_std, amax_std]
    ], headers, floatfmt='.4f')
print(table)
# %% [markdown]
# We may want to just transform how multobjective optimisation problem into a more straightforward
# problem by using a linear combination of objective functions.
# Both parameters have similar ranges in the pareto front so
# in this case so lets give them equal weight.

# %% 

# combine into a single objective function
lcobjective = (mean_objective + std_objective)/2
ax = plt.gca()
rtw._data_image(ax, dc, lcobjective, annotate=True, annotate_form=rtw.anform['lvel.mean'])
ax.set_title("Combined objective function")
plt.show()

# reading off good values for both parameters
eye_values = [[1.0, 1.0], [3.5, 2.0], [7.0, 5.0]]
eye_table = tabulate(eye_values, dc.pnames, floatfmt='.2f')
print('Hand picking some of the parameters from across the parameter space.')
print(eye_table)
# %%

# retrieve simulation index and path
eye_idx = [dc.find_index(xy) for xy in eye_values]
eye_dir = [dc.dircube[tuple(i)] for i in eye_idx]


# load linearized velocities
# for i, eye
lsimvel = []
for i, _ in enumerate(eye_idx):
    simdir = eye_dir[i]
    print('loading data from ', simdir)
    simdata = os.path.join(simdir, "data/")
    trs = readtrack.trackset(ddir=simdata)
    for tr in trs:
        tr._clean_bad_dt()
    ltrs = [_fj.linearize(tr) for tr in trs]
    track_vel = [twanalyse._inst_vel(tr) for tr in ltrs]
    simvel = np.concatenate(track_vel)
    # simvel = twutils.trim_tail(simvel, 0.05)
    lsimvel.append(simvel)

# %%
# check the bounds
table = []
for i, _ in enumerate(eye_idx):
    simvel = lsimvel[i]
    row = [os.path.basename(os.path.normpath(eye_dir[i]))]
    row.extend(np.quantile(simvel, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist())
    table.append(row)
table.append(['Fanjin', *np.quantile(allvel, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()])
print(tabulate(table, headers=['simulation path', 'min', '1st', '2nd', '3rd', 'max'], floatfmt='.4f'))

# %%
# We notice immediately from printing the 1st, 2nd and 3rd quantiles that these distributions
# are not all that similar. Simulated tracks in this dataset spend a large portion of the time
# idling at close to 0 velocity. This is even after linearising the trajectory. (Worth checking again)

# %%

ax = plt.gca()
plt.style.use(plotutils.get_style('jupyter'))
handle = []
label = []
xlims = (0, 0.1)
use_hist = False
for i, simvel in enumerate(lsimvel):
    outd = plotutils.ax_kdeplot(ax, simvel, xlims=xlims, hist=use_hist)
    handle.append(outd['handle'])
    label.append("({},{}) = ({:4.2f},{:4.2f})".format(*dc.prettynames(), *eye_values[i]))

hstyle = {'bins':100, 'density':True, 'alpha':0.2, 'rwidth':0.9}
ax.hist(allvel, range=xlims, **hstyle)
# print(hhandle)
# handle.append(hhandle)
# outd = plotutils.ax_kdeplot(ax, allvel, xlims=xlims, hist=use_hist)
# label.append('FanJin')
# handle.append(outd['handle'])
ax.set_xlabel(r'velocity $\mu m/s$')
ax.set_ylabel('P')
ax.legend(handle, label)
ax.set_xlim(xlims)
plt.show()

# %% [markdown]
# These are probability distributions and should all have the same area. Most likely 
# the kernal we use to compute probability density is not reflected at x = 0 but this library
# doesn't give the option to change that.
# It's always important to plot a straightforward histogram.
# %%

ax = plt.gca()
plt.style.use(plotutils.get_style('jupyter'))
handle = []
label = []
xlims = (0, 0.1)
use_hist = True
for i, simvel in enumerate(lsimvel):
    outd = plotutils.ax_kdeplot(ax, simvel, xlims=xlims, hist=use_hist)
    handle.append(outd['handle'])
    label.append("({},{}) = ({:4.2f},{:4.2f})".format(*dc.prettynames(), *eye_values[i]))
hstyle = {'bins':100, 'density':True, 'alpha':0.2}
ax.hist(allvel, range=xlims, **hstyle)
ax.legend(handle, label)
ax.set_xlim(xlims)
ax.set_ylabel('P')
ax.set_xlabel(r'velocity $\mu m/s$')
plt.show()

# %% [markdown]
# A histogram shows the problem with simulated tracks spending large amounts of time 
# stationary. It's been a while so I do need to test the code again for bugs and 
# before I analyse the trajectories to understand why this happens.
# it's also worth noting that standard deviation of velocity velocity is not very useful here 
# because the simulated distribution is so skewed.

# %%
# as Jure suggests, throw away the low velocity data so that we can see the high velocity distribution
low_threshold = 0.25
simvel = lsimvel[1]
pilivar, k_spawn = eye_values[1]

def labeller(names, values):
    return "({},{}) = ({:4.2f},{:4.2f})".format(*names, *values)
label = labeller(dc.prettynames(), (pilivar, k_spawn))
sim_high_vel = simvel[simvel > low_threshold]
print('cut out velocity < ', low_threshold)
print('which leaves {:4.2f}% of the data'.format(sim_high_vel.size/simvel.size))
ax = plt.gca()
print('velocity (min, max) = ', np.min(sim_high_vel), np.max(sim_high_vel))
outd = plotutils.ax_kdeplot(ax, sim_high_vel, res=100, hist=True)
handle = outd['handle']
ax.legend([handle], [label])
ax.set_title(r'only veloctity \textless {} ({:4.3f}\%)'.format(low_threshold, sim_high_vel.size/simvel.size))
# ax.set_xlim(xmin=sim_high_vel.min())
ax.set_ylabel('P')
ax.set_xlabel(r'velocity $\mu m/s$')
plt.show()

# %%
# do the same for fanjin data
def trim(allvel):
    s_allvel = np.sort(allvel)
    print('sorting finished')
    xlim = (0.25,2.5)
    bottomidx = np.searchsorted(s_allvel,  xlim[0])
    topidx = np.searchsorted(s_allvel,  xlim[1])
    trim_allvel = s_allvel[bottomidx:topidx]
    print(bottomidx/allvel.size)
    return trim_allvel
trim_allvel = trim(allvel)

# %%

ax = plt.gca()
hstyle = {'bins':100}
ax.hist(trim_allvel, **hstyle)
ax.set_xlim((0,2.5))
ax.set_ylabel('P')
ax.set_xlabel(r'velocity $\mu m/s$')
plt.show()

# outd = plotutils.ax_kdeplot(ax, allvel, xlims=xlims, hist=use_hist)
# no bump around 0.5

# %% 
# just for kicks lets do walking subset as well otherwise we might miss it
walking_idx, walking_fltrs = _fj.slicehelper.load_linearized_trs('default_walking_list', debug)
walking_allvel = np.concatenate([twanalyse._inst_vel(tr) for tr in walking_fltrs])
trim_allvel = trim(walking_allvel)

# %% 
ax = plt.gca()
hstyle = {'bins':100, 'rwidth':0.85}
ax.hist(trim_allvel, **hstyle)
ax.set_xlim((0,2.5))
ax.set_ylabel('P')
ax.set_xlabel(r'velocity $\mu m/s$')
plt.show()

# %% [markdown]
# No bumps in walking or crawling subsets.
# so what are we looking at?
# 1. Fanjin bacteria are using short pili//short pili contact time leading to translations << 0.12\mu m
# 2. Fanjin bacteria large \<nbound\> so the velocity and translation distance is highly contstrained
# 3. Fanjin bacteria pili do not have retraction speed ~0.5\mu m/s, perhaps 
# they have an effective retraction speed which is determined by some unconsidered factor (friction?)
# 4. something else ???

# %% 
# plot n fastest trajectories
print('cleaning up {} nan values'.format(np.count_nonzero(np.isnan(fjtrack_mean))))
# 3 nan values, why? 
fjtrack_mean[np.isnan(fjtrack_mean)] = 0
sort_vel_idx = np.argsort(fjtrack_mean)
top_vel = fjtrack_mean[sort_vel_idx[-10:]]
if verbose:
    for track_id in sort_vel_idx[-10:]:
        print('track id {} mean velocity {}'.format(track_id, fjtrack_mean[track_id]))
        tr = fltrs[track_id]
        ax = plt.gca()
        shapeplot.longtracks(ax,[tr])
        plt.show()
# %% 
# NOTE: move everyting below to classification.py =====>
# plot top 100 fastest trajectories sorted by velocity
# and save to file
vpdir = os.path.join(notename,'sort_fastest/')
if not os.path.exists(vpdir):
    os.makedirs(vpdir)
rule = os.path.join(vpdir,'track_{:04d}.png')
meta = {}
print('plotting rule ', rule)

ax = plt.gca()
for i, c_id in enumerate(reversed(sort_vel_idx[-100:])):
    ax.clear()
    tr = fltrs[c_id]
    mean_velocity = fjtrack_mean[c_id]
    track_id = idx[c_id]
    shapeplot.longtracks(ax, [tr])
    out = rule.format(i)
    plt.savefig(out)
    meta[i] = {'track_id':int(track_id),'c_id':int(c_id),'mean_velocity':mean_velocity}
    print(meta[i])
with open(os.path.join(vpdir,'meta.json'), 'w') as f:
    json.dump(meta,f,indent=1)
plt.close()

# %% 
# lets whitelist trajectories by eye which appear 
# to have similar characteristics to out candidate track
with open(join(vpdir,'candidates.list'), 'r') as f:
    _whitelist = list(map(int, f.read().split()))
with open(join(vpdir,'meta.json'), 'r') as f:
    meta = json.load(f)
whitelist = [meta[str(_i)]['c_id'] for _i in _whitelist]
track_whitelist = np.array([meta[str(_i)]['track_id'] for _i in _whitelist])
_fj.slicehelper.save('candidates_whitelist', track_whitelist)
print('whitelist', whitelist)


# top 100 trajectories by instantaneous velocity
_fj.slicehelper.save('top_plus', track_whitelist)



# %% 
# plot the step velocity distributions for the whitelist trajectories
dpdir = join(notename,'distribution/')
if not os.path.exists(dpdir):
    os.makedirs(dpdir)
rule = join(dpdir, 'lin_vel_{:04d}.png')
print('plotting rule ', rule)
ax = plt.gca()
for i, c_id in enumerate(whitelist):
    ax.clear()
    ax.set_ylabel('P')
    ax.set_xlabel(r'velocity $\mu m s^{-1}$')
    tr = fltrs[c_id]
    # twanalyse._ltr_vel(ax,[tr])
    # plotutils.ax_kdeplot(ax, vel, res=40, hist=True, xlims=(0,0.5))
    vel = np.linalg.norm(tr.get_step_velocity(),axis=1)
    hstyle = {'alpha':0.25, 'bins':40, 'range':(0,0.8),'density':True}
    ax.hist(vel, **hstyle)
    ax.set_ylabel('P')
    ax.set_xlabel(r'velocity $\mu m s^{-1}$')
    out = rule.format(_whitelist[i])
    plt.tight_layout()
    plt.savefig(out)
plt.close()

# %% 
# candidate
track_id = 2352
data_id = idx[track_id]
print()
print('data idx', data_id) # 2924
print()

# %% 
# draw candidate
candidate_track = _fj.trackload([data_id])[0]
ax = plt.gca()
shapeplot.ltdraw(ax, [candidate_track], sample=30)

# %% 
# mixed walking/crawling candidate
candidate_track = _fj.trackload([1279])[0]
ax = plt.gca()
_ = shapeplot.ltdraw(ax, [candidate_track], sample=60)


# %%
# velocity distribution of this candidate track
velocity = fjtrack_vel[track_id]
print('mean velocity ', np.mean(velocity))
ax = plt.gca()
plotutils.ax_kdeplot(ax, velocity, res=100, xlims=(0.0,1.5), hist=True)
plt.savefig('velocity_profiles/candidate_vel.png')
plt.show()

# %%
# linearised velocity distribution for canadidate track
ltcandidate = _fj.linearize(candidate_track)
ax = plt.gca()
twanalyse._ltr_vel(ax, [ltcandidate])
plt.show()

# %%
if work:
    savefile = 'plots/animate_linearised_{:04d}.mp4'.format(data_id)
    linearised_candidate = _fj.linearize(candidate_track)
    if not os.path.exists(savefile):
        print()
        print('animating track {} and saving at {}'.format(data_id, savefile))
        twanimation.outline(plt.gcf(), [linearised_candidate], sample=10, savefile=savefile)
        plt.clf()


# %% [markdown]
# pretty smooth fast trajectory
# seems to be able to translate sideways short distances while keeping orientation
# a feat usually only possible by the action of multiple pili to keep the orientation
# investigate this with simulation
# no fast/slow behaviour I can see, velocity is peaked around 0.1 but decays up to values 
# above 0.5 

# %% 
# what about the walking distribution
def _load_subset_speed():
    distrib = {}
    for name, ltrs in _fj.load_subsets().items():
        distrib[name] = np.concatenate([ltr.get_step_speed() for ltr in ltrs])
    return distrib
ref_vel = _load_subset_speed()
_vel = ref_vel["walking"]
# uncomment 2 lines to compare with unfiltered walking list
# _, all_walking = _fj.slicehelper.load_linearized_trs("default_walking_list")
# _vel = np.concatenate([ltr.get_step_speed() for ltr in all_walking])

import seaborn as sns
vel = ref_vel["walking"]
xlim = (0, np.quantile(vel, 0.98))
# ax =  sns.histplot(_vel, binrange=xlim) 
# walking velocity distribution is almost identical to crawling

# %% 
# load simulated crawling data
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r'\usepackage{booktabs}'
# %% 
#
simdata_collect = [
    "/home/dan/usb_twitching/run/two_parameter_model/two_parameters/pilivar_0004.00000_k_spawn_01.00000",
    "/home/dan/usb_twitching/run/new/angle_smoothed/range_pbrf/anchor_angle_smoothing_fraction_01.000",
    "/home/dan/usb_twitching/run/new/angle_smoothed/range_pbrf/anchor_angle_smoothing_fraction_00.250"
]
default = ["k_ext_off", "dwell_time", "pilivar",  "anchor_angle_smoothing_fraction", "k_spawn"]
xlabel = r"linearised speed $(\mu m\, s^{-1})$"
def plot_sim_with_table(ax, simdata_example):
    # parameters
    args = parameters.thisread(directory=simdata_example, par=default)
    row = [str(args.pget(name)) for name in par]
    col = [[x] for x in row]
    df = pd.DataFrame({"parameter":par, "value":row})
    #
    simtrs = readtrack.trackset(ddir=join(simdata_example, "data/"))
    simltrs = [_fj.linearize(tr) for tr in simtrs]

    sim_vel = np.concatenate([ltr.get_step_speed() for ltr in simltrs])
    sns.histplot(sim_vel, binrange=xlim, ax=ax, stat="density")
    ax.grid(False)
    ax.text(.5,.5, df.to_latex().replace('\n', ' '),
        transform=ax.transAxes, fontsize=20)
    ax.set_xlabel(xlabel)

# %% 
fig, axes = plt.subplots(2,2, figsize=(2*12,2*6))

ax = axes[0,0]
sns.histplot(ref_vel["walking"], binrange=xlim, ax=ax, stat="density") 
ax.set_xlabel(xlabel)
ax.text(0.5, 0.5, "Fanjin walking data", 
    transform=ax.transAxes, fontsize=24)
ax.grid(False)

twanalyse.plot_sim_with_table(axes[0,1], simdata_collect[0])
twanalyse.plot_sim_with_table(axes[1,0], simdata_collect[1])
twanalyse.plot_sim_with_table(axes[1,1], simdata_collect[2])

fig.tight_layout()

plt.savefig("/home/dan/usb_twitching/notes/sensitivity/compare_walking_velocity.png")

# %% [markdown]
# Walking velocity distribution is similar to simulated crawling data with no anchor constraint

# %% 
# plot fanjin linearised velocity




