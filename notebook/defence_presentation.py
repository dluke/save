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
# figure for defence presentation plots, lets try and make them fast using existing codes

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
norm = np.linalg.norm
import pili.support as support

import shapeplot
import readtrack
import readmat
import readmat
import pili
from pili.support import make_get, get_array
import pili.support as support
import _fj
import fjanalysis
import shapeplot


output = "/home/dan/usb_twitching/defense/ims"
writing_draw_dir = "/home/dan/usb_twitching/writing/draw"
settings = {"dpi": 300, "bbox_inches": 'tight', "transparent": True}

def save_figure(path, svg=False, settings=settings):
	_path = copy(path)
	if not path.endswith('.png'):
		path += '.png'
	print('save figure to ', path)
	plt.savefig(path, **settings)
	if svg:
		path = _path + '.svg'
		print('save figure to ', path)
		plt.savefig(path, **settings)


plotting = False

# %% 
#! 

# !EXP
subsetdata = _fj.load_subsets()
top = subsetdata["top"]
walking = subsetdata["walking"]

# !SIM
# simabc
# udir = join(sim4d["simdir"], sim4d["lookup"][0][prime_accepted.index[0]])
udir = '/home/dan/usb_twitching/run/825bd8f/cluster/mc4d/_u_aE9On8FI'
_trs = readtrack.trackset(ddir=join(udir, 'data/'))

# !SIM WALKING
udir = '/home/dan/usb_twitching/run/825bd8f/cluster/mc4d_walking/_u_fSNU1FWD'
walking_track = readtrack.trackset(ddir=join(udir, 'data/'))[0]

# %%
#! plotting setup
wstyle = dict(lw=3, alpha=1.0)

def to_origin(track):
	track['x'] -= track['x'][0]
	track['y'] -= track['y'][0]
	return track

# %%

#! plot simulated walking
walktr = _fj.linearize( readtrack.TrackLike(walking_track._track[0:10001]) )
x, y = walktr['x'], walktr['y']
print('duration', walktr.size//10)
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(x,y, **wstyle)
ax.axis(False)
ax.set_aspect("equal")

pt = np.array([0, 1.5])
WIDTH = 1.0
ax.plot([pt[0],pt[0]+WIDTH], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)


#! plot experiment walking

fig, ax = plt.subplots(figsize=(6,6))
index = 109
# Record Index: 2, 17, 27, 41, 59, 77, 85, 109 (!), 148
print('index', index)
walktr = to_origin(walking[index])
print('duration', walktr.size//10)
x, y = walktr['x'], walktr['y']
ax.plot(x,y, **wstyle)
ax.axis(False)
ax.set_aspect("equal")

pt = np.array([0, 1.5])
ax.plot([pt[0],pt[0]+WIDTH], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)

# %%
# ! combine onto one plot

c1 = "#4C92C3"
c2 = "#92C34C"

fig, ax = plt.subplots(figsize=(6,12))
ax.axis(False)
ax.set_aspect("equal")

walktr = _fj.linearize( readtrack.TrackLike(walking_track._track[0:8700]) )
print('duration', walktr.get_duration())
x, y = walktr['x'], walktr['y']
ax.plot(x,y, color=c1, **wstyle)

index = 17
walktr = to_origin(walking[index])
print('duration', walktr.get_duration())
x, y = walktr['x'], walktr['y']
vertical_shift = -2.0
horizontal_shift = -3.0
ax.plot(x + horizontal_shift, y + vertical_shift, color=c2, **wstyle)

pt = np.array([1.0, 0])
ax.plot([pt[0],pt[0]+WIDTH], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)

save_figure(join(writing_draw_dir, "walking_leading_pole"), svg=True)

# %%

# %%
# !plot simulated

lstyle = dict(lw=4, alpha=0.8)

def x_align(tr):
	#! align trajectory with the x_axis
	tr = tr.copy()
	x, y = tr['x'], tr['y']
	principle = np.array([x[-1] - x[0], y[-1] - y[0]])
	principle /= norm(principle)
	e_x = np.array([1,0])
	theta = np.sign(np.cross(principle, e_x)) * np.arccos(np.dot(e_x, principle))
	R = support.rotation_matrix_2d(theta)
	# shift and then rotate
	xm, ym = x - np.mean(x), y - np.mean(y)
	xy = np.column_stack([xm, ym])
	xyr = np.array([np.dot(R, xy[i]) for i in range(len(xy))])
	xyr -= xyr[0]
	tr['x'], tr['y'] = xyr.T
	return tr

index = 1
r_extr = _fj.linearize( readtrack.TrackLike(_trs[index]._track[:2001]) )
# _extr = to_origin(_fj.linearize( readtrack.TrackLike(_trs[index]._track[2001:4001]) ))

STEP = 2000


data_00 = np.array_split(_trs[0]._track, 20000//3333)
data_01 = np.array_split(_trs[1]._track, 20000//3333)

extr_list = [
	to_origin(_fj.linearize( readtrack.TrackLike(data_01[0]) )),
	to_origin(_fj.linearize( readtrack.TrackLike(data_00[1]) )),
	to_origin(_fj.linearize( readtrack.TrackLike(data_00[2]) )),
	to_origin(_fj.linearize( readtrack.TrackLike(data_00[3]) )),
	to_origin(_fj.linearize( readtrack.TrackLike(data_00[4]) ))
]


fig, ax = plt.subplots(figsize=(12,6))

# color = iter(support.color_list)
# color = iter(["#052CCC", "#058FCC", '#05CC42'])
# color = iter(["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"])
color_list =  ["#264653", "#2A9D8F", "#319632", "#F4A261", "#E76F51"]
color = iter(color_list)



SHIFT = 0.5
y_shift = 0
for _extr in extr_list:
	_extr = x_align(_extr)
	h1, = ax.plot(_extr['x'], _extr['y']-y_shift, color=next(color), **lstyle)
	y_shift += SHIFT

ax.axis(False)
ax.set_aspect('equal')

pt = np.array([0, 1.5])
ax.plot([pt[0],pt[0]+WIDTH], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)


save_figure(join(writing_draw_dir, "simulated_leading_pole"), svg=True)

if plotting:
	save_figure(join(output, 'example_simulated'))

# %%
# -5, -41, -53, -54
_extr = top[-6]

STEP = len(data_00[0])
print('STEP', STEP)

extr_list = [
	x_align(readtrack.TrackLike(top[-6]._track[:STEP])),
	x_align(readtrack.TrackLike(top[-7]._track[:STEP])),
	x_align(readtrack.TrackLike(top[-11]._track[:STEP])),
	x_align(readtrack.TrackLike(top[-14]._track[:STEP])),
	x_align(readtrack.TrackLike(top[-2]._track[:STEP]))
]

fig, ax = plt.subplots(figsize=(12,6))
_lstyle = lstyle.copy()

color = iter(color_list)

y_shift = 0
for _extr in extr_list:
	ax.plot(_extr['x'], _extr['y']-y_shift, color=next(color), **_lstyle)
	y_shift += SHIFT

ax.axis(False)
ax.set_aspect('equal')

# pt = np.array([_extr['x'][0], _extr['y'][0]-1])
pt = np.array([0, 1.5])
ax.plot([pt[0],pt[0]+WIDTH], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)

save_figure(join(writing_draw_dir, "top_leading_pole"), svg=True)

if plotting:
	save_figure(join(output, 'example_fanjin'))

# %%

def plot_track(ax, track, lw=3):
	ax.plot(track['x'], track['y'], lw=lw)
	ax.axis(False)
	ax.set_aspect('equal')

# plot_track(walking[6])
# plot_track(walking[17])

fig, ax = plt.subplots(figsize=(12,4))
track = walking[58]
plot_track(ax, track)

pt = np.array([track['x'][0], track['y'][0]-1])
width = -1.0
ax.plot([pt[0],pt[0]+width], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)

save_figure(join(output, 'example_walking'))

# %%

fig, ax = plt.subplots(figsize=(12,4))
track = top[20].cut(0,2000)

plot_track(ax, track, lw=6)

pt = np.array([track['x'][0], track['y'][0]-1])
width = 1.0
ax.plot([pt[0],pt[0]+width], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)

save_figure(join(output, 'example_crawling'))


# %%

np.argmax([len(track) for track in top])
# track = top[43]
candidate = _fj.trackload([2924])[0]
track = candidate
# candidate = _fj.trackload([2924])[0]
fig, ax = plt.subplots(figsize=(12,4))
plot_track(ax, track)

pt = np.array([track['x'][0], track['y'][0]-1])
width = 1.0
ax.plot([pt[0],pt[0]+width], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)


save_figure(join(output, 'candidate'))

fig, ax = plt.subplots(figsize=(12,4))
short = track.cut(200,400)
plot_track(ax, short)

pt = np.array([short['x'][0], short['y'][0]-1])
width = 1.0
ax.plot([pt[0],pt[0]+width], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)

save_figure(join(output, 'candidate_short'))



# %%

lincandidate = _fj.lintrackload([2924])[0]
track = lincandidate

fig, ax = plt.subplots(figsize=(12,4))
short = track.cut(1500, 1900)
plot_track(ax, short)


pt = np.array([short['x'][0], short['y'][0]-1])
width = 1.0
ax.plot([pt[0],pt[0]+width], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)

save_figure(join(output, 'candidate_part_01'))

fig, ax = plt.subplots(figsize=(12,4))
short = track.cut(750,1200)
plot_track(ax, short)

pt = np.array([short['x'][0], short['y'][0]-1])
width = 1.0
ax.plot([pt[0],pt[0]+width], [pt[1],pt[1]], linewidth=4, c='black', alpha=0.8)

save_figure(join(output, 'candidate_part_02'))



