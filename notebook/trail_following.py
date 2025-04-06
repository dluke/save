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
# INVESTIGATE trail following by checking 
# 1. whether trajectories overlap
# 2. if trajectories overlap, does this influence the direction of the bacteria

# %% 
import sys, os
import copy
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import scipy.stats
from copy import copy, deepcopy

import shapeplot
import readtrack
import readmat
import readmat
import pili
from pili.support import make_get, get_array
import pili.support as support
import _fj
import fjanalysis
import matdef
import shapeplot

import thesis.publication as thesis

import pili.publication as publication

# %% 

data_dir = 'trail_following'
codestyle = {"font.size": 24, "ytick.labelsize": 22, "xtick.labelsize": 22, "axes.labelsize" : 24, "legend.frameon" : False, "xtick.direction": 'in', "ytick.direction": 'in'} 
work = False

# %% 

load_fj = False

if load_fj:
	# !start by identifying which trajectory comes from which experiment
	experiment_ddirs = _fj.ddirs
	experiment_list = []
	for path in experiment_ddirs:
		expt = readmat.ExpTracks.usefj([path], False, tracktype=matdef.DENOISED)
		experiment_list.append(expt)

# %% 
idx, trs = _fj.slicehelper.load_trs("all")
len(trs)

# %% 
if load_fj:
	break_at = np.array([experiment.ntracks for experiment in experiment_list])
else:
	break_at = np.array([785, 594, 611, 798, 326])
experiment_index = np.split(idx, np.cumsum(break_at)[:-1])

# %% 
fig, ax = plt.subplots(figsize=(10,10))
one_experiment = experiment_index[0]
shapeplot.longtracks(ax, [trs[i] for i in one_experiment])

# %% 

# plotting helper
def _show_image(image, style={}):
	fig, ax = plt.subplots(figsize=(10,10))
	ax.imshow(image, origin='lower', **style)

# %% 
explore_radius = 0.5
# sample_radius = 3
sample_radius = 1.5
pix_size = 0.06

limits = readtrack.get_limits([trs[i] for i in one_experiment])

def get_size(limits):
	lx, ly = limits
	return np.array([lx[1]-lx[0], ly[1]-ly[0]])

def expand_limits(limits, delta):
	lx, ly = limits
	return [(lx[0]-delta, lx[1]+delta), (ly[0]-delta, ly[1]+delta)]

limits = expand_limits(limits, max(explore_radius, sample_radius))
field_size = get_size(limits)
pix_shape = 2*(field_size/pix_size/2).round().astype(int) 
lx, ly = limits
center = np.array([lx[0]+field_size[0]/2, ly[0]+field_size[1]/2])
justified_center = (center//pix_size) * pix_size
field_size = pix_size * pix_shape
field_at = justified_center - field_size/2
field_rect = (field_at, field_size)

print('limits', limits)
print('field_size ', field_size)
print('pix_size', pix_shape)
print('field_at', field_at)

# %% 

class Image(object):

	def __init__(self, pix_shape, pix_size=0.06):
		self.explored = np.zeros(pix_shape)
		self.pix_shape = pix_shape
		self.pix_size = pix_size
		self.field_at = field_at

	def write(self, path):
		np.savetxt(path, self.explored)

	def pt_to_index(self, at):
		return ( (at-field_at)/self.pix_size ).astype(int)

	def _get_square_idx(self, at, radius):
		sweep = np.array([radius, radius])
		return self.pt_to_index(at - sweep), self.pt_to_index(at + sweep)

	def fill_square(self, at, radius):
		xm, xp = self._get_square_idx(at, radius)
		xa, ya = xm
		xb, yb = xp
		if not all(np.array([xa,ya,xb,yb]) >= 0):
			print(at, np.array([xa,ya,xb,yb]))
			sys.exit()
		# print(xa, xb, ya, yb)
		self.explored[xa:xb,ya:yb] = 1.0

	def get_patch(self, radius):
		size = 2*(radius/self.pix_size) - 1
		a, b = -size*pix_size/2, size*pix_size/2
		n = int(np.ceil((b-a)/self.pix_size))
		if n % 2 == 0:
			n += 1
		X = np.linspace(a, b, n).reshape(1,-1)
		patch = np.sqrt(X**2 + X.T**2) 
		one = patch<=radius
		zero = patch>radius
		patch[one] = 1.0
		patch[zero] = 0.0
		return patch
		
	def fill_patch(self, at, patch, fill_type='&'):
		ix, iy = self.pt_to_index(at)
		n = (patch.shape[0]-1)//2
		if fill_type == '&':
			location = self.explored[ix-n:ix+n+1,iy-n:iy+n+1] 
			self.explored[ix-n:ix+n+1,iy-n:iy+n+1] = np.logical_or(location, patch)
		elif fill_type == 'sum':
			self.explored[ix-n:ix+n+1,iy-n:iy+n+1] += patch


	def explored_fraction(self):
		return self.explored.sum()/self.explored.size

	def copy(self):
		new = Image(self.pix_shape, self.pix_size)
		new.explored = self.explored.copy()
		return new


image = Image(pix_shape)
r = 1.5
plt.imshow( image.get_patch(r).T, origin='lower' )
image.get_patch(r).T.shape

# %%

# !load the crawling/walking datasets and compute their overlap with this psl visit map
subset_idx = _fj.load_subset_idx()
top_idx = subset_idx['top']
top_idx.sort()

break_top_idx = np.split(top_idx, np.searchsorted(top_idx, np.cumsum(break_at)[:-1]))
top_idx_one = break_top_idx[0]
top_idx_one 


# %% 
# blank image
blank = Image(pix_shape)

#! trajectories start at different times, we need to sort them
trs_one = [trs[i] for i in one_experiment]

def sort_by_start_time(index_list):
	lst = sorted( [(trs[track_index]["time"][0], track_index) for track_index in index_list], key=lambda x: x[0])
	return [s[1] for s in lst]

trs_one_index = sort_by_start_time(one_experiment)


DT = 0.1
class ImageGen(object):

	def __init__(self, 
			sorted_index_list, 
			max_time=None,
			fill_radius = 0.5,
			fill_type = '&'
			):
		self.r_index_list = copy(sorted_index_list)[::-1]
		self.blank = Image(pix_shape)
		self.curr_time = 0.1
		self.max_time = max_time if max_time is not None else 1999
		self.fill_radius = fill_radius
		self.patch = self.blank.get_patch(fill_radius)
		self.at = None
		self.n_iter = 0
		self.fill_type = fill_type

		self.active_list = []
		self.finished_list = []

	def _update_active_list(self, time):
		while self.r_index_list and trs[self.r_index_list[-1]]['time'][0] <= time:
			self.active_list.append( self.r_index_list.pop() )

	def __iter__(self):
		return self

	def __next__(self):
		# print('time', self.curr_time)
		if self.curr_time >= self.max_time:
			raise StopIteration

		# keep track of which trajectories are active at this time point
		self._update_active_list(self.curr_time)
		at = None
		for track_index in self.active_list:
			tr = trs[track_index]
			time = tr['time']
			# get the image index
			time_index = int( (self.curr_time - time[0])/DT )
			if time_index >= tr.size:
				self.active_list.remove(track_index)
				continue
			at = np.array([tr['x'][time_index], tr['y'][time_index]])
			# self.blank.fill_square(at, self.fill_radius)
			self.blank.fill_patch(at, self.patch, fill_type=self.fill_type)
		self.curr_time += DT
		self.n_iter += 1
		self.at = at
		return self.blank

	def get_last_at(self):
		return self.at
	
# %%
# 
#! generate image at regular intervals
gen = ImageGen(trs_one_index, max_time=None, fill_type='sum')
# gen = ImageGen(trs_one_index, max_time=100, fill_type='sum')

record_explored = []
record_image = [gen.blank]
with support.Timer():
	for image in gen:
		record_explored.append(image.explored_fraction())
		if gen.n_iter % 1000 == 0:
			print('saving at time', gen.curr_time)
			record_image.append(image.copy())
final_image = image

# %%
# _show_image(final_image.explored.T, style=dict(cmap='Greens'))

image = final_image.explored.T
style = {}

with mpl.rc_context(thesis.basestyle):
	fig, ax = plt.subplots(figsize=(10,6))

	_image = image + 1
	_image = np.clip(_image, _image.min(), 2*10**4)
	cmap = mpl.cm.Greens
	norm = mpl.colors.LogNorm(vmin=_image.min(), vmax=_image.max())

	ax.imshow(_image, origin='lower', norm=norm, cmap=cmap, **style)
	ax.axis('off')
	bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label="", shrink=.8)
	bar.ax.set_title("count", fontsize=24)

publication.save_figure('experiment_one_trails')
#! save the final ?

# %%
# ! do we need the same figure but normalised


# %%
# now as a test lets compute one track with a larger radius
def get_interval(tr):
	t = tr.get_start_time()
	return (t, t+tr.get_duration())

def generate_overlap(test_idx, index_list):
	end_time = get_interval(trs[test_idx])[1]
	print("end_time for track index {} is {} s".format(test_idx, end_time))

	exclude_one_idx = index_list.copy()
	exclude_one_idx.remove(test_idx)

	gen = ImageGen(exclude_one_idx, max_time=end_time)
	single_gen = ImageGen([test_idx], max_time=end_time, fill_radius= sample_radius)

	overlap = Image(pix_shape)
	for image, single in zip(gen, single_gen):

		patch = single.get_patch(sample_radius)
		n = (patch.shape[0]-1)//2

		at = single_gen.get_last_at()
		if at is None: continue
		ix, iy = single.pt_to_index(at)

		# view the image at this location
		view = image.explored[ix-n:ix+n+1,iy-n:iy+n+1] 
		overlap_patch = np.logical_and(view, patch)
		overlap.fill_patch(at, overlap_patch)

	return gen, single_gen, overlap

test = True
if test:
	test_idx = top_idx_one[7]
	print('test track', test_idx)
	gen, single_gen, overlap = generate_overlap(test_idx, trs_one_index)

	_show_image(gen.blank.explored.T)
	_show_image(single_gen.blank.explored.T)
	_show_image(overlap.explored.T)

	# compute the intersecting fraction
	overlap_fraction = overlap.explored.sum() / single_gen.blank.explored.sum()
	print('overlap fraction', overlap_fraction)

# %%
len(trs_one_index)

# %%
# ! now do it for every cell in the list of interest

if not os.path.exists(data_dir):
	os.mkdir(data_dir)

def runner(i, track_index, index_list):
	print('follow trail of track {:.1f}\n'.format( track_index ))
	gen, single_gen, overlap = generate_overlap(track_index, index_list)
	gen_path, single_path, overlap_path = [join(data_dir, '{}_{:04d}.npy'.format(prefix, track_index)) for prefix in ['gen', 'single', 'overlap']]
	def write(item, path):
		print('writing to ', path)
		item.write(path)
	write(gen.blank, gen_path)
	write(single_gen.blank, single_path)
	write(overlap, overlap_path)
	
# jobs = [(0, top_idx_one[7], sort_by_start_time(experiment_index[0]) )]
# support.parallel_run(runner, jobs)

if work:
	for k, experiment in enumerate(break_top_idx):
		with support.Timer():
			jobs = [(i, track_index, sort_by_start_time(experiment_index[k])) for (i, track_index) in enumerate(experiment)]
			support.parallel_run(runner, jobs)


# %%

def load_track_index(track_index):
	paths = [join(data_dir, '{}_{:04d}.npy'.format(prefix, track_index)) for prefix in ['gen', 'single', 'overlap']]
	def loadtxt(path):
		print('loading data at', path)
		return np.loadtxt(path)
	return [loadtxt(path) for path in paths]

def compute_overlap_fraction(single, overlap):
	return overlap.sum()/single.sum()



index = top_idx_one[7]
print("index", index)
image, single, overlap = load_track_index(index)
print('overlap', compute_overlap_fraction(single, overlap) )

_show_image(image.T)
_show_image(single.T)
_show_image(overlap.T)

# %%

# !load everything

def load_one_experiment(index_list):
	overlap_fraction = []
	for track_index in index_list:
		print('loading data for track ', track_index)
		image, single, overlap = load_track_index(track_index)
		fraction = compute_overlap_fraction(single, overlap)
		overlap_fraction.append(fraction)
	return overlap_fraction

overlap_fraction = []
for index_list in break_top_idx:
	overlaps = load_one_experiment(index_list)
	overlap_fraction.extend(overlaps)

# %%
# %%
#! save meta data
overlap_fraction = np.array(overlap_fraction)
overlap_data = np.column_stack([top_idx, overlap_fraction])
def savetxt(path, data):
	print("saving to ", path)
	np.savetxt(path, data)
savetxt(join(data_dir, "overlap_data.npy"), overlap_data)

# %%
#! load meta data
overlap_data = np.loadtxt(join(data_dir, "overlap_data.npy"))

# %%

#! overlap fraction distribution
overlap_fraction = overlap_data[:,1]

fig, ax = plt.subplots(figsize=(4,4))
xlim = (0,1.0)
sns.histplot(overlap_fraction, binrange=xlim, bins=10)
ax.set_xlabel("overlap fraction")

# %%
print('median/mean overlap fraction', np.median(overlap_fraction), np.mean(overlap_fraction))

# %%
#! load the statistics for these trajectories
ldata = fjanalysis.load_summary()

# %%
#! overlap fraction vs start-time
def get_track(index):
	return trs[index]
start_time = np.array([get_track(index)['time'][0] for index in top_idx])

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
mstyle = dict(marker='o', color='w', markerfacecolor=defcolor[0], markersize=13)

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	# mstyle = dict(s=80, marker='o', facecolor='none', edgecolor='blue')
	ax.plot(start_time, overlap_fraction, linestyle='none', **mstyle)
	ax.set_xlabel("initial time (s)")
	ax.set_ylabel("overlap fraction")
	ax.set_ylim(-0.1,1.1)

publication.save_figure("overlap_fraction_vs_initial_time")

# %%
# ! plot statistics 

top_ldata = []
for index_list in break_top_idx:
	one_ldata = [ldata[idx] for idx in index_list]
	top_ldata.extend(one_ldata)


with mpl.rc_context(thesis.texstyle):
	fig, axes = plt.subplots(2,2, figsize=(8,8), layout='constrained')
	axes = axes.flatten()

	objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
	for i, name in enumerate(objectives):
		ax = axes[i]	
		depdata = get_array(make_get(name), top_ldata)
		# fig, ax = plt.subplots(figsize=(4,4))
		ax.plot(overlap_fraction, depdata, linestyle='none', **mstyle)
		ax.set_xlabel('overlap fraction')
		ax.set_ylabel(name)
		ax.set_xlim(-0.1,1.1)

# %%
# ! we need to exclude some extreme outliers from our crawling subset
qhat_outlier_idx = np.argwhere( get_array(make_get('qhat.estimate'), top_ldata) < 0 ).ravel()[0]
track_index = top_idx[qhat_outlier_idx]
# ! the track is mostly fine but does some sort of horrendous oscillation at the end
# ax = plt.gca()
# shapeplot.longtracks(ax, [trs[track_index]], mark_endpoints=True)

# %%
#! now look at extreme values of ahat
ahat_outlier_idx = np.argwhere( get_array(make_get('ahat.estimate'), top_ldata) > 0.3 ).ravel()
for idx in ahat_outlier_idx:
	track_index = top_idx[idx]
	plt.figure()
	shapeplot.longtracks(plt.gca(), [trs[track_index]])

#! remove

# %%
# ! there is one extreme value of deviation angle statistics as well
var_theta_outlier_idx = np.argwhere(get_array(make_get('deviation.var'), top_ldata) > 3.0).ravel()[0]


# %%
#! mark these tracks for removal
to_remove = list(set([top_idx[qhat_outlier_idx], *[top_idx[idx] for idx in ahat_outlier_idx]]))
clean_top_idx = [idx for idx in top_idx if idx not in to_remove]
selector = [idx for idx in range(len(top_idx)) if idx not in [qhat_outlier_idx, *ahat_outlier_idx]]
clean_overlap_data = overlap_data[selector]
clean_ldata = [top_ldata[i] for i in selector]

# %%

clean_overlap_fraction = clean_overlap_data[:,1]
for name in objectives:
	depdata = get_array(make_get(name), clean_ldata)
	r, pvalue = scipy.stats.pearsonr(clean_overlap_fraction, depdata)
	print(f'{name} {r:.3f} {pvalue:.3f}')


# %%
# ! sort crawling trajectories by their overlap fraction and see if their statistics 
# ! are influenced.
# plot the clean data

pretty_name = [r'$\langle u \rangle$~(\textmu m/s)', r'$Var({\theta_d})$', r'$ \hat{q} $', r'$ \hat{a} $']
pretty_name = [r'$\langle u \rangle$~(\textmu m/s)', r'$Var({\theta_d})$', r'persistence $\hat{q}$', r'activity $\hat{a}$']

with mpl.rc_context(thesis.texstyle):
	fig, axes = plt.subplots(2,2, figsize=(8.8,7.5), layout='constrained', sharex=True)
	axes = axes.flatten()

	objectives = ['lvel.mean', 'deviation.var', 'qhat.estimate', 'ahat.estimate']
	for i, name in enumerate(objectives):
		ax = axes[i]	
		depdata = get_array(make_get(name), clean_ldata)
		# fig, ax = plt.subplots(figsize=(4,4))
		ax.plot(clean_overlap_fraction, depdata, linestyle='none', **mstyle)
		# ax.set_xlabel('overlap fraction')
		ax.set_ylabel(pretty_name[i])
		ax.set_xlim(-0.1,1.1)
		ax.yaxis.set_major_locator(plt.MaxNLocator(5))
	
		ax.set_ylim((0,None))

axes[-1].set_xlabel("overlap fraction")
axes[-2].set_xlabel("overlap fraction")

axes[1].set_ylim((0,3))
axes[2].set_ylim((0,None))
axes[3].set_ylim((0,None))

publication.save_figure("overlap_fraction_scatter")


# %%
# ! the same for walking


