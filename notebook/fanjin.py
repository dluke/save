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
# lets follow Fanjin a little more closely
# `/home/dan/usb_twitching/library/core/pnas_Wong_2011_slingshot_motion_observation.pdf`
# the purpose of this is to get a better understanding of analysing
# the trajectory so that we can fully understand how our simulated data
# differs from the experimental data

# %% 
import os
import sys
import collections
import numpy as np
import _fj
import matplotlib.pyplot as plt
import matplotlib as mpl
import command
import readtrack
import twanalyse 
import scipy.stats
from tqdm import tqdm

import seaborn as sns

# %% 
verbose = False
notename = 'fanjin'
notedir = 'fanjin/'
if not os.path.exists(notedir):
	os.makedirs(notedir)
join = os.path.join
def _savefig(name):
	d = os.path.dirname(name)
	if d:
		if not os.path.exists(d):
			os.makedirs(d)
	save = join(notedir,name)
	print("saving to ", save)
	plt.savefig(save)
# %% 
# ------------------------------------------------------------
idx, ltrs = _fj.slicehelper.load_linearized_trs('all')
# %% 
crawling_idx = _fj.slicehelper.load('default_crawling_list')
crawling = [ltrs[i] for i in crawling_idx]
candidate_idx = 2924
candidate = ltrs[candidate_idx]
whitelist = _fj.slicehelper.load('candidates_whitelist')
whitelist_tr = [ltrs[i] for i in whitelist]
# ------------------------------------------------------------

# %% 
# ------------------------------------------------------------
ax = plt.gca()
def plot_candidate(ax, track):
	time = track['time']
	velocity = track.get_head_v()
	track_speed = np.linalg.norm(velocity, axis=1)
	ax.plot(time[1:]-time[0], track_speed)
	ax.set_xlabel('time')
	ax.set_ylabel('$\mu m/s$')
# plot_candidate(ax, ltrs[10])
plot_candidate(ax, candidate)
plt.show()

# %% [markdown]
# these are linearised tracks, 
# lets check that displacements add up to 0.12 $\mu m$


# %% 
# plot displacement angles, body axis angles 
# for individual linearised trajectories

allpolar = twanalyse.allpolar
plotpolar = twanalyse.plotpolar
def plotpolar_with_axes(polardata):
	fig, axes = plt.subplots(3, 2, subplot_kw=dict(polar=True), 
		figsize=(8,12))
	plotpolar(axes, polardata)


# %%  
# polar plots for the candidate track only
polardata = allpolar([candidate])
plotpolar_with_axes(polardata)
_savefig("polar_candidate.png")

# %%  
# polar plots for whitelist 
polardata = allpolar(whitelist_tr)
desc = scipy.stats.describe(polardata.deviation)
plotpolar_with_axes(polardata)
_savefig("polar_whitelist.png")

# %%  
# polar plots for whitelist 

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
twanalyse.polar_hist(ax, polardata.deviation)
ax.set_title(r"$\cos^{-1}{\left(\vec{\hat{b}}\cdot\vec{\hat{u}}\right)}$", 
	fontsize=28)
plt.tight_layout()

# %%  
print(desc)
print('std = ', np.std(polardata.deviation))

# %%  
# crawling data
polardata = allpolar(crawling)
plotpolar_with_axes(polardata)
_savefig("polar_crawling.png")

# %%  
# all data
polardata = allpolar(ltrs)
plotpolar_with_axes(polardata)
_savefig("polar_all.png")

# %%  
# velocity > 0.3 (25%) ???
# is this a bug in my code or due outliers in the 'all' data
# lets make a walking list with a mean velocity threshold (?)
# see localdir for the removed outliers
localdir = 'vel_outlier'

def velocity_outliers(idx, ltrs, minvel=0.0, maxvel=1.0):
	meanvel = np.array([np.mean(np.linalg.norm(ltr.get_step_velocity(),axis=1)) for ltr in ltrs])
	print(np.isnan(meanvel).shape)
	print(idx.shape)
	nan_idx = idx[np.isnan(meanvel)]
	left_idx = idx[meanvel < minvel]
	right_idx = idx[meanvel > maxvel]
	return left_idx, right_idx, nan_idx
	
left, right, nan = velocity_outliers(idx, ltrs, maxvel=1.0)
bad_idx = np.union1d(right, nan)
l, r, n = left.size, right.size, nan.size
print('left\t{:4d}\tright{:4d}\tnan{:4d}'.format(l, r, n))
print('so there are {:d} candidates to remove'.format(bad_idx.size))

if verbose:
	import shapeplot
	for track_id in right:
		ltrack = ltrs[track_id]
		speed = np.linalg.norm(ltrack.get_step_velocity(), axis=1)
		print('fast step proportion', npcount(speed > 0.3)/speed.size)
		ax = plt.gca()
		shapeplot.longtracks(ax,[ltrack])
		name = join(localdir,'lt_{:04d}.png')
		_savefig(name.format(track_id))
		ax.clear()
	plt.close()

safeall = np.delete(idx, bad_idx)

# %%

polardata = allpolar([ltrs[i] for i in safeall])
plotpolar_with_axes(polardata)
_savefig("polar_all.png")
# and the value did change from 25% to 21% but its still huge
# compared to the crawling set, need more work to see if 
# high speed individuals in walking set are well resolved or not.



# %%
# * candidate trajectory, all polar plots are oriented towards direction of motion
# * whitelist, slow actions are polarised, fast actions have large components in forward direction
#   and also at 90 degrees
# * all tracks, the deviation angle for the fast mode are now dominated by
#   90 degree rotations, so these are associated with slower moving trajectories

# %%
# ------------------------------------------------------------
# lets do the same on simulation dataset
simdir = "/home/dan/usb_twitching/run/new/angle_smoothed/range_pbrf/anchor_angle_smoothing_fraction_00.250"
with command.chdir(simdir):
	simtrs = readtrack.trackset()
sim_ltrs = [_fj.linearize(tr) for tr in simtrs]
simtr = sim_ltrs[0]

plot_candidate(plt.gca(), simtr.part(0,2000))

# %%  
# simulated data 
polardata = allpolar(sim_ltrs)
plotpolar_with_axes(polardata)

# %% [markdown]
# which is similar to candidate. but note that there is not
#  much to distinguish fast and slow. 

# %%
# what about using a different velocity thresold?
# can we see a difference between fast and slow then?
quantiles = [0.5, 0.7, 0.9, 0.95]
actdata = twanalyse.actions(sim_ltrs)
def fraction(x, threshold):
	npcount = np.count_nonzero
	return npcount(x>threshold)/x.size

N = len(quantiles)
fig, axes = plt.subplots(N, 2, subplot_kw=dict(polar=True), 
	figsize=(8,4*N))
for i, q in enumerate(quantiles):
	quant = np.quantile(actdata['velocity'], q)
	row = axes[i]
	pdata = twanalyse.allpolar(sim_ltrs, quant)
	fastidx = actdata['velocity'] > quant
	twanalyse.polar_dev(row, pdata.deviation, fastidx)
	row[0].set_title(r'velocity $>$ {:3.1f} ({:d}\%)'.format(
		quant, int(100*q)
	))
	plt.tight_layout()
	
# %%
candidate_action = twanalyse.actions([candidate])

# %%
# plot candidate displacements
ax =plt.gca()
bins = np.arange(0,max(candidate_action['dx']),0.03)
sns.histplot(candidate_action['dx'], bins=bins)
ax.set_xlabel('action displacement')
ax.axvline(0.12, linestyle='--', color='k')

# %%
# plot candidate action dt
ax =plt.gca()
ax.set_ylabel('P')
ax.set_xlabel('action time (s)')
sns.histplot(candidate_action['dt'])

# %%
actions = twanalyse.actions
plot_actiondata = twanalyse.plot_actiondata
candidate_action = actions([candidate])
traction = actions(crawling)

# %%
plot_actiondata(candidate_action)
_savefig('candidate_actions.png')
plt.show()

# %%
# all data 
if verbose:
	plot_actiondata(traction)
	_savefig('actions.png')
	plt.show()

# %%
# [markdown]
# find out where the 0.1/0.2 second fast actions are hiding, look at those trajectories by eye.

# %%
# plot 2d histogram for actions A,A+1

def action_corr(actions):
	# first bin all the actions but keep track of which bin they belong to
	vel = actions['velocity']
	vel = twanalyse.trim_limits(vel, [0.004,16])
	bins = np.geomspace(0.004,33,16+1,True)
	with np.printoptions(precision=3, suppress=True):
		print('bins', bins)
	bincount = np.zeros(bins.size-1,dtype=int) # one less bin than edge
	binmap = [] 
	nbin = []
	count2d = np.zeros((bins.size-1,bins.size-1), dtype=int)
	for i in range(vel.size-1):
		v = vel[i]
		v_n = vel[i+1]
		bin_idx = np.searchsorted(bins, v)
		nbin_idx = np.searchsorted(bins, v_n)
		binmap.append(bin_idx)
		nbin.append(nbin_idx)
		bincount[bin_idx] += 1
		count2d[bin_idx][nbin_idx] += 1
	
	X = bins
	Y = bins
	return X, Y, count2d

X, Y, count2d = action_corr(candidate_action)
def plot_c2d(X, Y, count2d):
	ax = plt.gca()
	ax.set_xscale('log')
	ax.set_yscale('log')
	cmap = mpl.cm.get_cmap()
	cmlist = [cmap(c) for c in np.linspace(0,1,100,True)]
	cm = mpl.colors.LinearSegmentedColormap.from_list('adj', cmlist)
	Im = np.ma.masked_where(count2d == 0, count2d)
	# norm = mpl.colors.Normalize(count2d.min(), count2d.max())
	norm = mpl.colors.LogNorm(1.0, count2d.max())
	pos = plt.pcolormesh(X, Y, Im, cmap=cm, norm=norm)
	plt.colorbar(pos)
	ax.set_aspect('equal')
plot_c2d(X, Y, count2d)

# %%
X, Y, c2d = action_corr(traction)
plot_c2d(X, Y, c2d)


# %% [markdown]
# Following Fanjin, we considered each linerised step as one action and 
# analysed the deviation angle between actions. 
# We split the actions into fast and slow using a velocity threshold of 0.3 microns/s
# which was chosen by eye from inspecting the peaks in the velocity profile
# Its unclear from fanjin whether consective fast and slow steps are 
# combined to form individual actions. We choose not to do this.
# Our results are mostly consistent with Fanjin although some unexplained
# differences do exist.


# %%
# for target simulation, /run/825bd8f/target/t0, approximately 10% of displacements by a single pilus
# fall above the step threshold, d_step = 0.12
# (bear in mind there is no time constraint for these displacements)

