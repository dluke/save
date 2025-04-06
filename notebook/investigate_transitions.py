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
# investigate the experimental data for evidence of walking/crawling transitions or lack thereof
# As usual this means looking at surface projected aspect ratio 

# %% 
import sys, os
import copy
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns 

import pili
from pili import support
from pili.support import make_get, get_array
import readtrack
import command
import _fj
import fjanalysis
import twanalyse

import pili.publication as pub

import thesis.publication as thesis

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']

# %% 
verbose = False

# %% 

idx, trs = _fj.slicehelper.load_original_trs('all')

ldata = fjanalysis.load_summary()
lvel = get_array(make_get('lvel.mean'), ldata)


# %% 
# load thesis classifications
from glob import glob

paths = glob(join(pili.root, 'notebook/thesis_classification', '*.npy'))
print('found paths\n', '\n'.join(paths))

def strip_name(path):
	return os.path.split(os.path.splitext(path)[0])[-1]

def load_classifications(paths):
	classes = []
	for path in paths:
		pair = (strip_name(path), np.loadtxt(path).astype(int))
		classes.append(pair)
	return dict(classes)

idx_list = load_classifications(paths)

#! thesis classifications
# walking_idx, crawling_idx = idx_list['walking'], idx_list['crawling']

#! paper classifications
walking_idx, crawling_idx = _fj.slicehelper.load("pure_walking"), _fj.slicehelper.load("candidates_whitelist") 
walking_trs, crawling_trs = [trs[i] for i in walking_idx], [trs[i] for i in crawling_idx]

print('using {} walking and {} crawling trajectories'.format(walking_idx.size, crawling_idx.size))


# %% 
# ! compute and save the coarse graining data
def _coarse_graining(arr, framewindow=100):
	if arr.size <= framewindow:
		return np.nan
	frh = framewindow//2
	sbasis = np.arange(frh, arr.size-frh, 1, dtype=int)
	return np.array([np.mean(arr[j-frh:j+frh-1]) for j in sbasis])

# TODO use non-overlapping intervals because its faster?
# def coarsen(lptr, n):
# 	N = lptr.M
# 	lptr.x = np.array([lptr.x[i:i+n].mean() for i in range(0, N-n)])
# 	lptr.y = np.array([lptr.y[i:i+n].mean() for i in range(0, N-n)])
# 	lptr.M = len(lptr.x)
# 	return lptr

with support.Timer():
	w_aspect = np.array([_coarse_graining(tr['length']/tr['width']) for tr in walking_trs], dtype=object)
	c_aspect = np.array([_coarse_graining(tr['length']/tr['width']) for tr in crawling_trs], dtype=object)

#! save temporaries
# _w_aspect = w_aspect
# _c_aspect = c_aspect


# %% 
c_max_aspect = [np.max(data) for data in c_aspect]
w_max_aspect = [np.max(data) for data in w_aspect]

c_min_aspect = [np.min(data) for data in c_aspect]
w_min_aspect = [np.min(data) for data in w_aspect]

print('aspect range ', (np.min(c_min_aspect), np.max(c_max_aspect)), (np.min(w_min_aspect), np.max(w_max_aspect)))

# TODO alternatively sort by max aspect 
w_aspect_var = np.array([np.var(data) for data in w_aspect])
c_aspect_var = np.array([np.var(data) for data in c_aspect])

print('means', np.mean(c_aspect_var), np.mean(w_aspect_var))

# %% 
#! plot the variance in gamma_t distribution

with mpl.rc_context(thesis.texstyle):
	fig, ax  = plt.subplots(figsize=(4,4))
	violinstyle = dict(showmedians=False)
	ax.violinplot([w_aspect_var, c_aspect_var], **violinstyle)
	ax.set_xticks([1,2], labels=['walking', 'crawling'])
	ax.set_title(r"$\mathrm{Var}(\gamma_t)$")

# %% 

w_label = r"$b_{\mathrm{min}} < 1.6$ (walking)"
c_label = r"$b_{\mathrm{min}} > 1.6$ (crawling)"

#! don't like violinplot here, try histogram again
_mplstyle = {**thesis.texstyle, 'xtick.direction':'out'}
with mpl.rc_context(_mplstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	shstyle = dict(log_scale = True, alpha = 0.5, shrink=0.9)
	sns.histplot(w_aspect_var, color='b', ax=ax, label=w_label, **shstyle)
	sns.histplot(c_aspect_var, color='r', ax=ax, label=c_label, **shstyle)
	xlabel = r"$\mathrm{Var}(b_t)$"
	ax.set_xlabel(xlabel)
	ax.legend(loc=(1.04, 0.7))

	ax.yaxis.set_major_locator(plt.MaxNLocator(5))

	# arrow = dict(facecolor='black', width=1, shrink=0.05, headwidth=4)
	# ax.annotate('A', (0.1, 5), xytext=(1.04, 0.3), textcoords='axes fraction', arrowprops=arrow)

# ax.annotate(r'$\delta_{\mathrm{step}}$', (0.12, 10), xytext=(0.6, 0.5), textcoords='axes fraction', fontsize=16, arrowprops=dict(facecolor='black', width=1, shrink=0.05, headwidth=4))

pub.save_figure("variance_of_the_aspect_ratio")


# %% 
# ! the same idea but lets use a scatterplot with kmsd as the other axis
# get_array(make_get(''), ldata)
ldata[0].keys()

# from msddistrib.py
geomscaling = np.array([1,2,3,5,8,13,20,32,50,100,200,300,500])

def compute_msd(track, scaling=geomscaling):
	xy = track.get_head2d()
	x_, y_ = xy[0,0], xy[0,1]
	x, y = xy[:,0] - x_, xy[:,1] - y_ # subtract the starting position
	msd_n = []
	for i, window in enumerate(scaling):
		msd = (x[window:] - x[:-window])**2 +  (y[window:] - y[:-window])**2 
		msd_n.append(msd)
	return np.array([np.mean(msd) for msd in msd_n])

def fit_kmsd(basis, msd):
	basis = np.array(basis)
	msd = np.array(msd)
	select = basis > 10
	l_basis, l_msd = np.log(basis[select]), np.log(msd[select])
	p, cov = np.polyfit(l_basis, l_msd, deg=1, cov=True)
	return p[0]

msd_data = np.array([fit_kmsd(geomscaling, compute_msd(tr)) for tr in trs])

# %% 
# w_msd, c_msd = msd_data[walking_idx], msd_data[crawling_idx]

# with mpl.rc_context(thesis.texstyle):
# 	sstyle = dict(alpha=0.5)
# 	fig, ax = plt.subplots(figsize=(4,4))
# 	ax.scatter(w_msd, w_aspect_var, **sstyle)
# 	ax.scatter(c_msd, c_aspect_var, **sstyle)




# %% 
# sns.histplot(w_aspect_var)
# look at the crawling trajectories with high variance

c_sort_idx = np.argsort(c_aspect_var)
w_sort_idx= np.argsort(w_aspect_var)

c_sort_aspect = c_aspect[c_sort_idx]
w_sort_aspect = w_aspect[w_sort_idx]

# check sizes
sizes = np.array([data.size for data in w_sort_aspect])
len(w_aspect), (sizes > 2000).sum()

selector = sizes > 2000
select_w_aspect = [w_sort_aspect[i] for i in range(len(w_aspect)) if selector[i]]
c_selector = np.array([data.size for data in c_sort_aspect]) > 2000
select_c_aspect = [c_sort_aspect[i] for i in range(len(c_aspect)) if c_selector[i]]

c_select_idx = c_sort_idx[c_selector]


# %% 
lstyle = dict(lw=4)
if verbose:
	for data in select_w_aspect[::-1]:
	# for i in range(1,5):
		fig, ax = plt.subplots(figsize=(10,2))
		ax.plot(data, **lstyle)
		# ax.set_ylim(0.9, None)
		ax.set_ylim(0.9 * np.min(w_min_aspect), 1.1 * np.max(w_max_aspect))

# %% 
# %% 
cool = mpl.cm.get_cmap("cool")
winter = mpl.cm.get_cmap("winter")

# %% 
# %% 
#! choose data of similar length

median_idx = len(select_w_aspect)//2
# datalist = [select_w_aspect[-1], select_w_aspect[-4], select_w_aspect[0]]
datalist = list(reversed([*select_w_aspect[median_idx:median_idx+3], *select_w_aspect[-4:None]]))
color_space = np.linspace(0.5,1,len(datalist),False)
color = iter([cool(c) for c in color_space])

hstyle = dict(alpha=0.4, c='k')

def shifted_profiles(ax, datalist, size_limit = 3000, min_shift = 0.6, const_shift = 0.4):
	shift = 0
	shiftlist = []
	for i, data in enumerate(datalist):
		data = data[:size_limit]
		time = np.linspace(0, len(data)/10, len(data))
		# mx = np.max(data)-shift
		# mn = np.min(data)-shift
		ax.axhline(np.max(data)-shift, linestyle=(5, (10,3)), **hstyle)
		ax.axhline(np.min(data)-shift, linestyle=(5, (10,3)), **hstyle)

		ax.plot(time, data-shift, color=next(color), **lstyle)
		shiftlist.append(shift)
		if i < len(datalist)-1:
			shift += max(np.max(datalist[i+1]) - np.min(datalist[i+1]) + const_shift, min_shift)
			print(shift)

		if i == 3:
		   shift += 2

	return np.array(shiftlist)

with mpl.rc_context(thesis.basestyle):
	fig, ax = plt.subplots(figsize=(10,10))
	mindata = np.array([np.min(data) for data in datalist])
	maxdata = np.array([np.max(data) for data in datalist])
	shiftlist = shifted_profiles(ax, datalist)
	# ax.axis(False)
	# ax.yaxis.set_visible(False)
	data = datalist[0]
	ytickdata = [np.max(data), np.min(data)]
	ytickdata.extend(mindata[1:].tolist() - shiftlist[1:]) 
	labels = [r'$b_{\mathrm{max}} = %.1f$' % ytickdata[0], r'$b_{\mathrm{min}} = %.1f$' % ytickdata[1]] 
	labels.extend([r'$b_{\mathrm{min}} = %.1f$' % yt for yt in mindata[1:]])
	ax.yaxis.set_ticks(ytickdata, labels=labels)
	plt.draw()
	draw_ylim = ax.get_ylim()

	for name in ['left', 'top', 'right']:
		ax.spines[name].set_visible(False) 
	ax.set_xlabel('time (s)')

	# vertical line
	spinestyle = dict(c='k', lw=1)
	shift_min = mindata - shiftlist
	shift_max = maxdata - shiftlist
	break_index = 3
	delta = 0.3
	ax.plot([0,0], [draw_ylim[0], shift_max[break_index+1]+delta], **spinestyle)
	ax.plot([0,0], [shift_min[break_index]-delta, draw_ylim[-1]], **spinestyle)

	with mpl.rc_context({'path.sketch': (5, 13, 1)}):
		ax.plot([0,0], [shift_max[break_index+1]+delta, shift_min[break_index]-delta], **spinestyle)
	# zigzag(ax, (0))
	ax.set_ylim(draw_ylim)

	delta = 0.1
	x,y = 0-delta, mindata[0]-delta
	c_highlight = dict(alpha=0.2, color=defcolor[0])
	w_highlight = dict(alpha=0.2, color=defcolor[1])
	ws_highlight = dict(alpha=0.2, color=defcolor[2])

	c_patch = mpl.patches.Rectangle((x,y), 120, 3.1+2*delta, **c_highlight)
	ax.add_patch(c_patch)

	ws_patch = mpl.patches.Rectangle((140,y), 160, 3.1+2*delta, **ws_highlight)
	ax.add_patch(ws_patch)


	x,y = 190, mindata[1]-shiftlist[1]-delta
	c_patch = mpl.patches.Rectangle((x,y), 80, (maxdata[1]-mindata[1])+2*delta, **c_highlight)
	ax.add_patch(c_patch)

	w_patch = mpl.patches.Rectangle((0,y), 180,(maxdata[1]-mindata[1])+2*delta, **w_highlight)
	ax.add_patch(w_patch)

	ax.legend([c_patch, w_patch, ws_patch], ['crawling', 'walking', r'walking*'], loc=(1.10,0.77))
	# ax.annotate(r"$\gamma = \mathrm{length}/\mathrm{width}$", (0.5,0.5))
	ax.set_title(r"$b = \mathrm{length}/\mathrm{width}$")

	# draw patchs to indicate walking/crawling and uncertain regions

	arrow_x = -10
	y = np.max(data)
	textwidth = 50
	ax.annotate("$b$".format(y), (-25, mindata[0]+1.4))
	patch = mpl.patches.FancyArrowPatch((arrow_x,np.max(data)), (arrow_x,np.min(data)), arrowstyle='<|-|>', mutation_scale=20, lw=2)
	ax.add_patch(patch)


	ax = fig.add_axes((0.9, 0.0, 0.2, 0.9))
	ax.axis(False)
	c_arrow = "#696969"
	patch = mpl.patches.FancyArrowPatch((0.2,0.2), (0.2, 0.9), arrowstyle='-|>', mutation_scale=50, lw=8, facecolor=c_arrow, edgecolor=c_arrow)
	ax.add_patch(patch)
	ax.annotate(r"$\mathrm{Var}(b_t)$", (0.3,0.5), rotation=-90)
	# plt.tight_layout()

# TODO not happy with the figure .. split them into their own axes
#! (didn't do this before because sizing the axes is slightly annoying)


# %%
from matplotlib.ticker import StrMethodFormatter
color = iter([cool(c) for c in color_space])

with mpl.rc_context(thesis.texstyle):
	fig, axes = plt.subplots(8,1, sharex=True, sharey=True, figsize=(8,8))
	i = 0
	axes[4].axis(False)
	for data in datalist:
		data = data[:3000]
		ax = axes[i]
		ax.plot(0.1*np.arange(len(data)), data, color=next(color), **lstyle)

		ax.axhline(np.max(data), linestyle=(5, (10,3)), **hstyle)
		# ax.axhline(np.min(data), linestyle=(5, (10,3)), **hstyle)

		ax.set_ylim(0.9, 5.0)
		ax.set_yticks([1.0, 4.0])
		i += 1 if i != 3 else 2
		ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
		ax.tick_params(axis='y', which='major', labelsize=18)
	fig.supylabel("b = length/width")	
	ax.set_xlabel('time (s)')

	ax = fig.add_axes((0.9, 0.0, 0.2, 0.9))
	ax.axis(False)
	c_arrow = "#696969"
	patch = mpl.patches.FancyArrowPatch((0.2,0.2), (0.2, 0.9), arrowstyle='-|>', mutation_scale=40, lw=8, facecolor=c_arrow, edgecolor=c_arrow)
	ax.add_patch(patch)
	ax.annotate(r"$\mathrm{Var}(b_t)$", (0.3,0.5), rotation=-90)
	# plt.tight_layout()

pub.save_figure("aspect_ratio_walking_examples")

# %%
# 
color_space = np.linspace(0.5,1,len(datalist),False)
color = iter([winter(c) for c in color_space])

verbose = False
if verbose:
	for data in select_c_aspect[::-1]:
		fig, ax = plt.subplots(figsize=(10,2))
		lstyle = dict(lw=4)
		ax.plot(data, **lstyle)
		ax.set_ylim(0.9 * np.min(w_min_aspect), 1.1 * np.max(w_max_aspect))
else:
	# just the first two 
	idx = 40
	idx = len(select_c_aspect)//2 + 1
	median_data = select_c_aspect[idx]
	max_data = select_c_aspect[40]

	with mpl.rc_context(thesis.texstyle):
		fig, ax = plt.subplots(figsize=(10,2))
		lstyle = dict(lw=4, alpha=0.8)
		h1, = ax.plot(0.1*np.arange(median_data.size), median_data, color=winter(50), **lstyle)
		h2, = ax.plot(0.1*np.arange(max_data.size), max_data, color=winter(200), **lstyle)
		ax.set_ylim(0.9 * np.min(w_min_aspect), 1.1 * np.max(w_max_aspect))
		# ax.axhline(np.max(data), linestyle='--', **hstyle)
		# ax.axhline(np.min(data), linestyle='--', **hstyle)
		ax.set_xlabel('time (s)')
		ax.set_ylabel(r'$b_t$')
		ax.legend([h1, h2], [r'median $\mathrm{Var}(b_t)$', r'max $\mathrm{Var}(b_t)$'], loc=(1.04,0.4))

pub.save_figure("aspect_ratio_crawling_examples")
		
# %%
winter

# %%
# do the crawling trajectories with somewhat high variance in the aspect ratio
# idx = 40
# break contact with the surface?
track_idx = crawling_idx[c_select_idx[idx]]
c_aspect_var[c_select_idx[idx]]

# c_aspect_var[c_select_idx]
tr = trs[track_idx]
ltr = _fj.lintrackload([track_idx])[0]

fig, ax = plt.subplots(figsize=(10,2))
v = ltr.get_step_speed()
dt = ltr.get_step_dt()
ax.plot(ltr.step_idx[:-1], ltr.get_step_speed(), linestyle='none', marker='o')

# so
ltr.step_idx[1]
split_idx = np.searchsorted(ltr.step_idx, 500)

# t[:split_idx]
v[:split_idx].mean(), v[split_idx:].mean()

def get_qa(ltr):
	ld = twanalyse._qaparams([ltr])
	return ld['qhat']['estimate'], ld['ahat']['estimate']

left = _fj.linearize(tr.cut(0,500))
right = _fj.linearize(tr.cut(500,None))

# get_qa(ltr.cut(500,None))
get_qa(_fj.linearize(left)), get_qa(_fj.linearize(right))

#! persistence and velocity both slightly corrrelated with 


# %%