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
# analyse simulations on hexagonal packed sphere surface
#

# %%

import os
import numpy as np
import scipy.stats

import thesis.publication as thesis

import _fj
import pili
import rtw
import twanalyse
import readtrack

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


join = lambda *x: os.path.abspath(os.path.join(*x))
norm = np.linalg.norm
sem = scipy.stats.sem

import pili.support as support

# %%
rundir = join(pili.root, '../run')
# target = join(rundir, "2f9f469/vary_radius_long")
# target = join(rundir, "2f9f469/relax_anchor/vary_radius")
# target = join(rundir, "2f9f469/repeat_10/vary_radius")
target = join(rundir, "2f9f469/horizontal/repeat_10/vary_radius_3")

# load linear search data using this old structure
dc = rtw.DataCube(target=target)
track_data = dc.calculate(readtrack.trackset)




# %%
sphere_radius = dc.basis.ravel()

# %%

load_attractive = True

if load_attractive:
	# target = join(rundir, "2f9f469/relax_anchor/attr_vary_radius")
	target = join(rundir, "2f9f469/horizontal/repeat_10/attr/vary_radius_3")
	dc = rtw.DataCube(target=target)
	attr_track_data = dc.calculate(readtrack.trackset)
	attr_kmsd_list, attr_kmsd_list_sem = compute_kmsd(attr_track_data)

# %%
# %%
# TODO
# !plotting example trajectories with the hexgrid visible

import shapeplot


index = 0
track_list = attr_track_data[index]

def annotate_ruler(ax, pt, length=20, linewidth=4):
	ax.plot([pt[0],pt[0]+length], [pt[1],pt[1]], linewidth=linewidth, c='black', alpha=0.8)
	delta = 0.005
	# ax.text(pt[0]+length + delta + 0.005, pt[1]-delta-0.005, r"$0.1$\textmu m", fontsize=14)

def plot_tracks(ax, track_list, lstyle={}):

	for tr in track_list:
		ax.plot(tr['x'], tr['y'], **lstyle)
	
	ax.axis('off')


lstyle = {"lw":2.5}
with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(8,8))
	plot_tracks(ax, track_list, lstyle)
	annotate_ruler(ax, np.array([-50,50]))

thesis.save_figure("attr_hexsphere_tracks")

# %%

index = 3
print('R', sphere_radius[index])
track_list = attr_track_data[index]

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(8,8))
	plot_tracks(ax, track_list, lstyle)
	annotate_ruler(ax, np.array([-40,40]), length=10)
	shapeplot.draw_hex_grid(ax, sphere_radius[index], ax.get_xlim(), ax.get_ylim(), N=50, usepatches=False)

thesis.save_figure("attr_hexsphere_tracks_R_2")

# %%


index = 5
print('R', sphere_radius[index])
track_list = attr_track_data[index]

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(8,8))
	plot_tracks(ax, track_list, lstyle)
	annotate_ruler(ax, np.array([-10,21]), length=5)
	guidestyle = shapeplot._guidestyle.copy()
	guidestyle['s'] = 8
	shapeplot.draw_hex_grid(ax, sphere_radius[index], ax.get_xlim(), ax.get_ylim(), N=50, usepatches=False, guidestyle=guidestyle)

	ax.set_aspect("equal")

thesis.save_figure("attr_hexsphere_tracks_R_4")

# %%

# cut data after plotting
if load_attractive:
	attr_track_data = [[tr.cut_time(2000, 4000) for tr in track_list] for track_list in attr_track_data]

# %%
check_size = False
if check_size:
	for track_list in track_data:
		for tr in track_list:
			print(tr.size)
	
# %%
if check_size:
	for track_list in track_data:
		print(track_list)
		for tr in track_list:
			print( tr.cut_time(2000,4000).size )

# cut the data
track_data = [[tr.cut_time(2000, 4000) for tr in track_list] for track_list in track_data]

# %%

# calculate total distance travelled and end to end distance

def displacement_length(tr):
	x, y = tr['x'], tr['y']
	return np.sqrt((x[0]-x[-1])**2 + (y[0]-y[-1])**2)


def contour_length(tr):
	dx = tr.get_dx()
	return norm(dx, axis=1).sum()


# contour_length()
contour_length(track_data[0][0])
displacement_length(track_data[0][0])

c_datalist = [np.array([contour_length(tr) for tr in track_list])
			  for track_list in track_data]
h_datalist = [np.array([displacement_length(tr)
					   for tr in track_list]) for track_list in track_data]

c_list = [c.mean() for c in c_datalist]
c_list_std = [sem(c) for c in c_datalist]
h_list = [h.mean() for h in h_datalist]
h_list_std = [sem(h) for h in h_datalist]

attr_c_datalist = [np.array([contour_length(tr) for tr in track_list])
			  for track_list in attr_track_data]
attr_h_datalist = [np.array([displacement_length(tr)
					   for tr in track_list]) for track_list in attr_track_data]

attr_c_list = [c.mean() for c in attr_c_datalist]
attr_c_list_std = [sem(c) for c in attr_c_datalist]
attr_h_list = [h.mean() for h in attr_h_datalist]
attr_h_list_std = [sem(h) for h in attr_h_datalist]

style = dict(marker='D', capsize=4, markersize=5, linestyle='--')

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4, 4))

	ax.errorbar(sphere_radius, c_list, c_list_std, **style)

	ax.errorbar(sphere_radius, attr_c_list, attr_c_list_std, **style)

	ax.set_xlabel(r"sphere radius, $R$")
	# ax.set_ylabel("contour length")
	ax.set_ylabel(r"$C$ (\textmu m) ", fontsize=28)
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	plt.xscale("log")
	ax.set_xticks([0.25, 1.0, 4.0, 16.0])
	ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax.set_ylim(0,  None)

thesis.save_figure("horizontal_start_contour_length_hexsurface")

# hc = np.array(h_list)/np.array(c_list)

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4, 4))
	h1 = ax.errorbar(sphere_radius, h_list, h_list_std, **style)
	h2 = ax.errorbar(sphere_radius, attr_h_list, attr_h_list_std, **style)

	ax.set_xlabel(r"sphere radius, $R$")
	# ax.set_ylabel("end-to-end displacement")
	ax.set_ylabel(r"$h$ (\textmu m) ", fontsize=28)
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))

	handles = [h1,h2]
	labels = ["no attraction", "attractive"]
	ax.legend(handles, labels, loc=(0.60, 0.65), handlelength=1, framealpha=1.0, frameon=True, fontsize=20)


	plt.xscale("log")
	ax.set_xticks([0.25, 1.0, 4.0, 16.0])
	ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax.set_ylim(0,  None)

thesis.save_figure("horizontal_start_h_length_hexsurface")

# %%
# compute distance from start position as a time series

def timed_displacement_length(tr):
	x, y = tr['x'], tr['y']
	return np.sqrt((x[0]-x)**2 + (y[0]-y)**2)


def plot_total_displacement(ax, track_data):
	for track_list in track_data:
		for tr in track_list:
			ax.plot( timed_displacement_length(tr), lw=2, alpha=0.7)
			break
		
	ax.legend(labels=sphere_radius)

fig, ax = plt.subplots(figsize=(12,5))
plot_total_displacement(ax, track_data)
# plot_total_displacement(ax, attr_track_data)

# fig, ax = plt.subplots(figsize=(12,5))
# for track_list in track_data:
# 	for tr in track_list:
# 		ax.plot( tr['z'] - tr['z'][0], lw=2, alpha=0.7)
	
# ax.legend(labels=sphere_radius)



# %%
# TODO
# compute MSD
# run with surface attraction on and off

scaling = np.arange(2, 500).astype(int)


def msd(tr, scaling):
	xyz = tr.get_head()
	x, y, z = xyz.T
	msd_n = np.empty(scaling.size)
	for i, window in enumerate(scaling):
		sd = (x[window:] - x[:-window])**2 + (y[window:] - y[:-window])**2
		msd_n[i] = np.mean(sd)
	return msd_n


def msd_reduce(track_list, scaling):
	return np.stack([msd(tr, scaling) for tr in track_list]).mean(axis=0)


# SLOW
msd_list = [msd_reduce(track_list, scaling) for track_list in track_data]

def compute_kmsd(track_data):
	kmsd_data = [[twanalyse.kmsd(tr) for tr in track_list]
				 for track_list in track_data]
	kmsd_list = [np.mean(kmsd) for kmsd in kmsd_data]
	kmsd_list_sem = [sem(kmsd) for kmsd in kmsd_data]
	return kmsd_list, kmsd_list_sem

with support.Timer():
	kmsd_list, kmsd_list_sem = compute_kmsd(track_data)



# %%
# +ve surface interaction leads to high persistence, similar to plane
# 
# TODO check that the system has equilibrated for large sphere radius, R

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4, 4))
	lstyle = dict(lw=2)
	handles = []
	for msd_curve in msd_list:
		h, = ax.loglog(scaling * 0.1, msd_curve, **lstyle)
		handles.append(h)

	labels = ['R = {:.2f}'.format(R) for R in sphere_radius]
	ax.legend(handles, labels, loc=(1.04, -0.1))
	ax.set_xlabel(r'$\tau$ (s)')
	ax.set_ylabel(r'MSD($\tau$)')

# %%

with mpl.rc_context(thesis.texstyle):
	fig, ax = plt.subplots(figsize=(4, 4))

	style = dict(marker='D', capsize=4, markersize=5)

	ax.axhline(1.0, linestyle='--', alpha=0.4, c='k')

	def plot_kmsd(basis, kmsd_list, kmsd_list_sem, style):
		return ax.errorbar(basis, kmsd_list, kmsd_list_sem, **style)
	handles = []
	h1 = plot_kmsd(sphere_radius, kmsd_list, kmsd_list_sem, style)
	handles.append(h1)
	if load_attractive:
		h2 = plot_kmsd(sphere_radius, attr_kmsd_list, attr_kmsd_list_sem, style)
		handles.append(h2)
	labels = ["no attraction", "attractive"]

	ax.legend(handles, labels, loc=(0.60, 0.65), handlelength=1, framealpha=1.0, frameon=True, fontsize=20)

	plt.xscale("log")
	ax.set_xticks([0.25, 1.0, 4.0, 16.0])
	ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax.set_ylim(0, 2.0)
	ax.yaxis.set_major_locator(plt.MaxNLocator(4))
	ax.set_ylabel(r'$k_{\mathrm{MSD}}$')
	ax.set_xlabel(r'sphere radius, $R$')

thesis.save_figure("horizontal_start_kmsd_on_hexsphere")

# %%
# lets plot the angle relative to the surface


def surface_angle(tr):
	xyz = tr.get_head()
	x, y, z = xyz.T
	xy = np.sqrt(x**2 + y**2)
	angle = np.arctan2(z, xy)
	return angle

index = 4
print('index, R', index, sphere_radius[index])
# angle = surface_angle(track_data[index][0])
fig, ax = plt.subplots(figsize=(6,4))
for tr in attr_track_data[index]: 
	angle = surface_angle(tr)
	ax.plot(angle)
ax.set_ylim(0, np.pi/2)



