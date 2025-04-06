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
# preliminary summary statistics for walking/crawling data
# such as total rotation in xy plane (why not just use persistence?)
# aspect ratio distributions are shown, note that after removing short 
# trajectories the distribution shapes do not change
# note that in this analysis walking/crawling are are defined only 
# by aspect ratio

# for human identification of walking trajectories see annotate_walking.py


# %% 
import os
import sys
import numpy as np
import _fj
import matplotlib.pyplot as plt
import matplotlib as mpl
import command
import readtrack
import fjanalysis
import twanalyse
import stats
import rtw
join = os.path.join
norm = np.linalg.norm


# %%
notename = 'classification'
notedir = os.path.normpath(os.path.dirname(__file__))

# %%
idx, ltrs = _fj.slicehelper.load_linearized_trs('all')
_ , trs = _fj.slicehelper.load_trs('all')
walking_list = _fj.slicehelper.load('default_walking_list') 
crawling_list = _fj.slicehelper.load('default_crawling_list')

# %%
# load local analysis data output by fjanalysis.outliers
with command.chdir('../src/analysis/'):
    fjlocal = stats.load()
    # convert all lists to numpy array
    for k, v in fjlocal.items():
        if isinstance(v, list):
            fjlocal[k] = np.array(v)
print('fanjin local data')
print(list(fjlocal.keys()))
min_aspect_ratio = fjlocal['min_aspect_ratio']

# %%
# SETUP
# we are going to be generating a lot of subsets of the data, lets use dictionary
idxd = {}

# %%
# plotting styles
vlinestyle = {'color':'k', 'linestyle':'--'} 

# %%
# plot the number of steps in each track
track_nsteps = np.array([len(ltr.step_idx) for ltr in ltrs])
histstyle = {"rwidth":0.9}
def  plot_nsteps(track_nsteps):
    fig, axes = plt.subplots(1,2, figsize=(10,6))
    ax1, ax2 = axes
    ax1.hist(track_nsteps,range=(0,500), **histstyle)
    ax2.hist(track_nsteps,range=(500,np.max(track_nsteps)), **histstyle)
    # ax2.set_xticks([500,3000,5000])
    ax = fig.add_subplot(111, frameon=False)
    ax.grid(False)
    # hide tick and tick label of the big axis
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("Number of linearised steps")
    ax.set_ylabel("P")
print("Fanjin data")
print("step number distribution")
plot_nsteps(track_nsteps)

# %%
# before doing anything else lets delete some outliers which have low step count
low_nstep_threshold = 20
idxd["low_nstep_threshold"] = track_nsteps < low_nstep_threshold
count = np.sum(track_nsteps < low_nstep_threshold)
# '{:.2f}%'.format(100*float(count)/len(track_nsteps))
'{}/{} tracks are below the thresold'.format(count, len(track_nsteps))
def plot_aspect(min_aspect_ratio):
    # using idxd in global scope
    fig, axes = plt.subplots(1,2, figsize=(10,6), sharex=True)
    ax1, ax2 = axes
    n1, _bins, _patches = ax1.hist(min_aspect_ratio, 30)
    xticks = [1,3,5]
    for ax in axes:
        ax.set_xticks(xticks)
    min_aspect_filtered =  min_aspect_ratio[~idxd["low_nstep_threshold"]] 
    n2, _bins, _patches = ax2.hist(min_aspect_filtered, 30)
    ax1.set_ylabel('P')
    ax2.set_ylabel('P')
    ax1.set_xlabel('length/width')
    ax2.set_xlabel('length/width')
    ax1.set_title('{} tracks'.format(len(min_aspect_ratio)))
    ax2.set_title('{} tracks'.format(len(min_aspect_filtered)))
    ymax = 1.1*max(max(n1),max(n2))
    ax1.set_ylim((None,ymax))
    ax2.set_ylim((None,ymax))
    return fig, axes
print("min aspect ratio distibutions with short tracks subtracted")
_ = plot_aspect(min_aspect_ratio)
# the peak at 1.0 remains after filtering so we know that these have 
# significant step count
# There is no particular visible change in the distribution

# %%
# what about simple average aspect ratio?
print("mean aspect ratio")
print("vertical line shows minimum at approx 1.8")
aspect_ratio = np.array([np.mean(tr['length']/tr['width']) for tr in trs])
fig, axes = plot_aspect(aspect_ratio)
ax1, ax2 = axes
line = ax1.axvline(1.8, **vlinestyle)

# %%
# -------------------------------
# COLLECT UTILITY FUNCTIONS

# WARN: believe it or not body_length can be identically 0 in this dataset
# allow this function to return nan
def get_bodyxy(ltr):
    # Get step body axis projection
    step_idx = np.array(ltr.step_idx)
    timebase = ltr['time'][step_idx]
    headxy = ltr.get_head()[step_idx][:,:2]
    trailxy = ltr.get_trail()[step_idx][:,:2]
    bodyv = headxy - trailxy 
    # split this body vector into direction and length components
    body_length = np.sqrt(np.sum(bodyv**2,axis=1))
    with np.errstate(divide='ignore', invalid='ignore') as errstate:
        body_axis = bodyv/body_length[:,np.newaxis]
    return body_axis, body_length

# %%
# this is a good first step to separating crawling and walking behaviour
# but lets increase specificity by adding another calculation
# autocorrelation coefficient of the body axis
# lets start with our candidate tracjectory
candidate_idx = 2924
candidate_track = ltrs[2924]
body_axis, _ = get_bodyxy(candidate_track)
# autocorrelation

s = 1
corr = []
for s in range(1, len(body_axis)):
    bdot = np.sum(body_axis[s:,:]*body_axis[:-s,:], axis=1)
    corr.append(np.mean(bdot))
ax = plt.gca()
ax.plot(corr)
ax.set_ylim(0,1)
ax.set_ylabel("body axis auto correlation")
ax.set_xlabel("$\Delta step$")

# crawling trajectories typically have such high persistance that they don't change direction
# significantly on the experimental timescale
# lets consider a distance of 3 microns, similar to cell body length and max pili length
# we should be able to characterise two types of motion, one with correlation length << 5 
# and another with correlation length >> 5

# %%
# note that we need some characteristic walking trajectories that we checked by eye 
# new notebook: walking_candidate.py

# %%
# correlation is not the most convenient calculation here, lets define the
# the "total absolute rotation" and rotation per head travel distance
def total_rotation(ltr):
    # sum the rotation between each linearised step
    body_axis, body_length = get_bodyxy(ltr)
    # WARN: body_length can be 0 => body_axis can be NaN
    deviations = np.abs(np.arccos(np.sum(body_axis[1:]*body_axis[:-1], axis=1)))
    total_angle = np.nansum(deviations)
    return total_angle

def total_distance(ltr):
    dx = ltr.get_step_dx() # displacements
    dd = np.sqrt(np.sum(dx*dx, axis=1))
    total_distance = np.sum(dd)
    return total_distance

walking_candidate_idx = 33
print("walking candidate track ", walking_candidate_idx)
w_candidate = ltrs[walking_candidate_idx]
walking_speed = np.mean(w_candidate.get_step_speed())
print("speed {:.3f} microns/s".format(walking_speed))
trot = total_rotation(w_candidate)
tdist = total_distance(w_candidate)
print("total rotation {:.3f} radians".format(trot))
print("total head travel distance {:.3f} microns".format(tdist))
print("total rotation/distance {:.3f} radians/micron".format(trot/tdist))
print("total rotation/s {:.3f} radians/s".format(trot/w_candidate.get_duration()))
print()

candidate_track = ltrs[2924]
print("crawling candidate track ", 2924)
crawling_speed = np.mean(candidate_track.get_step_speed())
print("speed {:.3f} microns/s".format(crawling_speed))
trot = total_rotation(candidate_track)
tdist = total_distance(candidate_track)
print("total rotation {:.3f} radians".format(trot))
print("total head travel distance {:.3f} microns".format(tdist))
print("total rotation/distance {:.3f} radians/micron".format(trot/tdist))
print("total rotation/s {:.3f} radians/s".format(trot/candidate_track.get_duration()))

# %%
# quick summary function to apply ot the the whole dataset
# computes the body rotation in xy plane relative to distance travelled and time
import collections 
Rot = collections.namedtuple("Rot", ["total", "travel", "rpd", "rps"])
def rotation_summary(ltr):
    trot = total_rotation(ltr)
    travel = total_distance(ltr)
    duration = ltr.get_duration()
    return Rot(trot, travel, trot/travel, trot/duration)
rotdata = [rotation_summary(ltr) for ltr in ltrs]

# %%
# plot walking and crawling body rotation rates
a = [rotdata[idx].rpd for idx in idx]
warr = [rotdata[idx].rpd for idx in walking_list]
carr = [rotdata[idx].rpd for idx in crawling_list]
xlim = (0,4)
hstyle = {"alpha":0.5, "range":xlim, "density":True}
ax = plt.gca()
ax.hist(warr, bins=30, color='b', label="walking", **hstyle)
ax.hist(carr, bins=30, color='r', label="crawling", **hstyle)
# ax.hist(a, bins=60, color='r', label="", **hstyle)
ax.set_ylabel("P")
ax.set_xlabel("body rotation (radians/micron traveled)")
ax.legend()


# NOTE it might also be worth calculating the rotation of the start and end positions
# for crawling tracks -- we expect this to be much less than the total rotation of
# but can simulation explain why this is?

# %%
# simulation data
root = os.path.abspath(join(notedir, '../..'))
walking_sim = join(root,'run/ec44f05/walking/range_spawn')
dc = rtw.DataCube(target=walking_sim)
trcube = dc.autocalculate(readtrack.trackset, force=True)
print(dc.basis)
ltr  = _fj.linearize(trcube[4][0])
rot = rotation_summary(ltr)
print("simulated walking ")
print("total rotation/distance {:.3f} radians/micron".format(rot.rpd))
print("total rotation/s {:.3f} radians/s".format(rot.rps))

# simulation data
crawling_sim = join(root, 'run/two_parameter_model/two_parameters/pilivar_0004.00000_k_spawn_03.00000')
print(crawling_sim)
with command.chdir(crawling_sim):
    tr = readtrack.trackset()[0]
    ltr = _fj.linearize(tr)

rot = rotation_summary(ltr)
print("simulated crawling")
print("total rotation/distance {:.3f} radians/micron".format(rot.rpd))
print("total rotation/s {:.3f} radians/s".format(rot.rps))


# %%
