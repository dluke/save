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
# presentation plots

# %% 
import os
import json
import numpy as np
import scipy.stats
import pickle
join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sctml
import sctml.publication as pub
print("writing figures to", pub.writedir)

import pili
from pili import support
import _fj
import mdl
import pwlpartition
import pwltree

import fjanalysis
import pwlstats

from skimage.restoration import denoise_wavelet
from skimage.restoration import  estimate_sigma

# %% 
mplstyle = {"font.size": 20}
notename = "mdlpresent"

# %% 
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

plot_target = join(pwlstats.root, "impress/images")

target = join(plot_target, "candidate_whole.png")

print('saving to ', target)
plt.savefig(target, bbox_inches='tight')



# %% 

ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.30, 'color': defcolor[2], 'markersize': 12}
plot_target = join(pwlstats.root, "impress/images")



fig, ax = plt.subplots(figsize=(12,12))
ax.plot(short.x, short.y, label='data', **ptlkw)
ax.set_aspect("equal")
ax.set_axis_off()

ax.set_xlabel("x")
ax.set_ylabel("y")

target = join(plot_target, "candidate_data_for_drawing.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target, bbox_inches='tight')


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

# %% 
# use pwltree to generate some simple piecewise solutions for short data

def solve(data, initmodel, r):
	tsolver = pwltree.TreeSolver(data, overlap=True)
	tsolver.build_initial_tree(initmodel)
	tsolver.build_priority()
	tsolver.solve(pwltree.stop_at(r))
	return tsolver

sigma = pwlpartition.estimate_error(data.x, data.y)
r = 0.03

t_model = []
r_search = [0.01,0.03, 0.05,0.10]
for r in r_search:
	tsolver = solve(shortdata, wavemodel, r)
	tree_model = tsolver.get_model()
	t_model.append(tree_model)

# %% 

for i, tree_model in enumerate(t_model):
	r = r_search[i]
	with mpl.rc_context({'font.size': 16}):
		fig, ax = plt.subplots(figsize=(12,5))
		local_simple_plot(ax, tree_model, shortdata)
		ax.set_ylabel('y')
		ax.set_xlabel('x')
		ax.set_title('r = {:.2f}'.format(r))

	target = join(plot_target, "wavelet_candidate_treesolve_{:02d}.png".format(i))
	print('saving to ', target)
	plt.tight_layout()
	plt.savefig(target)

# %% 
#! plot with nice formatting
defense = "/home/dan/usb_twitching/defense/ims"

defcolor = plt.rcParams['axes.prop_cycle'].by_key()['color']

c2 = '#DA5025'
c3 = '#2AA1C6'
blue = c3

model_style = {"linestyle": '-', 'lw':2, 'alpha':1.0, 'color':c2, "marker":'D', 'markerfacecolor' : 'none', 'markeredgewidth':3, 'markersize': 16}
ptlkw = {"linestyle":'none', 'lw':1.5, "markersize" : 10, "marker":"o", "alpha":0.7, 'color':c3, 'markerfacecolor': 'none', 'markeredgewidth':2.0}

for i, tree_model in enumerate(t_model):
	r = r_search[i]
	print("r = ", r)
	with mpl.rc_context({'font.size': 16}):
		fig, ax = plt.subplots(figsize=(8,5))
		model = tree_model 
		data = shortdata

		ax.plot(data.x, data.y, **ptlkw)
		ax.plot(model.x, model.y, **model_style)
		ax.axis(False)
		ax.set_aspect('equal')
		plt.draw()
		ax.set_ylim(ax.get_ylim()[0]-0.1, ax.get_ylim()[-1]+0.1)
	#
		# ax.set_title('r = {:.2f}'.format(r))

	target = join(defense, "candidate_treesolve_{:02d}.png".format(i))
	print('saving to ', target)
	plt.tight_layout()
	plt.savefig(target, bbox_inches='tight', transparent=True)



# %% 

r = 0.055
tsolver = solve(shortdata, wavemodel, r)

# %% 
import itertools

def plot_partition(ax, model, data):

	color = itertools.cycle(['#FEC216', '#F85D66', '#75E76E'])
	time = model.get_time()
	split = np.split(data.get_n2(), time[:None])
	for sp in split:
		c = next(color)
		x, y = sp.T
		ax.plot(x, y, c=c, linestyle='none', marker='o')
	ax.set_aspect("equal")

	ax.set_xlabel('x')
	ax.set_ylabel('y')

with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots(figsize=(12,12))
	ax.set_axis_off()
	plot_partition(ax, tsolver.get_model(), shortdata)


plot_target = join(pwlstats.root, "impress/animation/")
target = join(plot_target, "candidatet_tree_color_partitions.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target, bbox_inches='tight')

# %% 

with mpl.rc_context(mplstyle):
	fig, ax = plt.subplots(figsize=(12,12))
	ax.set_axis_off()
	pwlpartition.simple_model_plot(ax, tsolver.get_model(), shortdata)

target = join(plot_target, "candidate_tree_final.png")
print('saving to ', target)
plt.tight_layout()
plt.savefig(target, bbox_inches='tight')


# %% 


# animate the tree solver
# 1. plot the data as a background
# 2. construct linear segments in rectangular boxes

fig, ax = plt.subplots(figsize=(8,5))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)


def init():
	pkw = ptlkw.copy()
	pkw["linestyle"] = 'none'
	ax.plot(short.x, short.y, label='data', **pkw)
	ax.set_aspect("equal")
	ax.set_axis_off()

	# pad axes
	pad = 0.1
	x, y = short.x, short.y
	ax.set_xlim(x.min()-pad, x.max()+pad)
	ax.set_ylim(y.min()-pad, y.max()+pad)


point_data = tsolver.point_data

def bounds(data):
	# return rect as [min, max], [min, max]
	x, y = data.T
	return [[x.min(), x.max()], [y.min(), y.max()]]

def rpatch(ax, lx, ly, delta=0.02):
	# plot rectangle patch
	lx.sort()
	ly.sort()
	width = lx[1] - lx[0] + 2*delta
	height = ly[1] - ly[0] + 2*delta
	xy = [lx[0] - delta, ly[0] - delta]
	patch = mpl.patches.Rectangle(xy, width, height, fill=True, alpha=0.1, linewidth=4.0, color="yellow")
	ax.add_patch(patch)
	return patch

def linear_model(m, theta, cx):
	return np.array([m*np.cos(theta) + cx, m*np.sin(theta)])

def segment_in_rect(theta, cx, rect):
	lx, ly = rect
	lxm, lxp = lx
	lym, lyp = ly
	
	#! unsafe divide by zero
	mm = max( (lxm - cx)/np.cos(theta), lym/np.sin(theta) )
	mp = min( (lxp - cx)/np.cos(theta), lyp/np.sin(theta) )

	return linear_model(mm, theta, cx), linear_model(mp, theta, cx)

import matplotlib.patheffects as pe

# lstyle = dict(color=c2, alpha=0.8, linewidth=6, path_effects=[pe.Stroke(linewidth=10, foreground='k', alpha=0.8), pe.Normal()] )
lstyle = dict(color=c2, alpha=0.8, linewidth=4, marker='D', markersize=12, markerfacecolor='none', markeredgewidth=1.5)

# for node in state.children:
def update(frame, artist={}, tsolver=tsolver):
	print('animate frame', frame)
	line = artist.get('line')
	# patch = artist.get('patch')
	history = tsolver.get_history()
	state = history[frame]["tree"]
	index = history[frame]["index"]

	# clear the lines and patch
	# if patch:
	# 	patch.remove()
	while line:
		line.pop().remove()

	for i, node in enumerate(state.children):

		if i == index:
			style = lstyle.copy()
			style["color"] = "#5FE000"
		else:
			style = lstyle.copy()


		fi, ti = node.f_index, node.t_index+1
		part = point_data[fi:ti]
		lx, ly = bounds(part)

		# expensive to recompute the fit here
		loss = tsolver.Loss(part)
		result = tsolver.linear_fit(loss)

		if result.fun > 0:
			theta, cx = result.x
			if abs(abs(theta) - np.pi/2) < 1e-1:
				continue
			pm, pt = segment_in_rect(theta, cx, [lx, ly])
			x1, y1 = pm
			x2, y2 = pt

			# if i == index:
			# 	artist['patch'] = rpatch(ax, [x1,x2], [y1,y2])
		else:
			# two points, no fitting
			x1, y1 = part[0]
			x2, y2 = part[-1]
		l, = ax.plot([x1,x2], [y1,y2], **style)
		line.append(l)

line = []
artist = {'line' : []}
init()
update(0, artist, tsolver)
update(-1, artist, tsolver)

savefile = join(plot_target, "candidate_last_frame.png")
print('saving to ', savefile)
plt.savefig(savefile)


# %%

from functools import partial
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8,3))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

n = len(tsolver.get_history())
# n = 10

# mpl.rcParams['nbagg.transparent'] = False


artist = {'line' : []}
ani = FuncAnimation(
	fig, partial(update, artist=artist, tsolver=tsolver),
	frames = range(n), 
	init_func=init, interval=200 
)


plot_target = join(pwlstats.root, "impress/animation/")
savefile = join(plot_target, "animate_candiate_recursion.mp4")

print(savefile)
ani.save(savefile, savefig_kwargs={'bbox_inches':'tight', 'transparent':False})
# fig
# plt.show()

# %%
# !draw a simple linear regression figure

import thesis.publication as publication

_style = ptlkw.copy()
_style['markersize'] = 8
with mpl.rc_context(publication.texstyle):
	fig, ax = plt.subplots(figsize=(4,4))
	t = np.arange(0.02,0.98,0.01)
	y = t + np.random.normal(scale=0.10, size=t.size)
	ax.plot(t, y, **_style)
	ax.set_xlabel('t')
	# ax.set_ylabel('y')
	fig.tight_layout()

plt.savefig(join(defense, "simple_lsq_regression.png"), transparent=True)


