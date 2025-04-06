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
# the purpose of this short notebook is just to examine how velocities are calculated and 
# why large velocities of 2 or 3 $\mu m/s$ are reported in analysis

# %% 
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import readtrack
import command
import pilush
import plotutils
import twutils
import _fj
import fjanalysis
import twanalyse
from tmos import base
from tabulate import tabulate
import stats
plotutils.default_style()

# %% 
# datapaths
testdata = "/home/dan/usb_twitching/run/c450a32/two_parameters_koch/pilivar_0050.00000_k_spawn_10.00000/testview/trackxy.dat"
tr = readtrack.Track(testdata)
print("loading data  at", testdata)
verbose = False

# load candidate
track_id = 2924
candidate = _fj.lintrackload([track_id])[0]

# %% 
# check that linearized velocity is computed correctly
# and as a result fjanalysis.lsummary() and twanalyse.summary()
# compute correct velocities
ltr = _fj.linearize(tr)
inst_speed = np.mean(twanalyse._inst_vel(tr))

lvel = stats.col_stats([ltr], 
    lambda tr: tr.get_step_speed(), 
    weight_f=lambda tr: tr.get_step_dt())

sd = twanalyse.observables([ltr])
sd["lvel"]["mean"]

head = ["instantaneous speed", "step speed", "check step speed", "check summary"]
data = [[np.mean(inst_speed), lvel["mean"], twanalyse.lvel(ltr), sd["lvel"]["mean"]]]
print(tabulate(data, headers=head))


print("candidate")
inst_speed = np.mean(twanalyse._inst_vel(candidate))
c_sd = twanalyse.observables([candidate])
lvel = c_sd["lvel"]["mean"]
inst_speed, lvel
# so candidate velocities are also almost identical

# %% 

# head velocity
def _displacement(tr):
    xy = np.column_stack([tr['x'], tr['y']])
    dxy = xy[1:] - xy[:-1]
    disp = np.linalg.norm(dxy,axis=1)
    dt = tr['time'][1:] - tr['time'][:-1]
    return xy, dxy, disp, dt
xy, dxy, disp, dt = _displacement(tr)
# vel =  / dt[:,np.newaxis]
print('timesteps are in range ', np.min(dt), np.max(dt))
print('displacements are in range ', np.min(disp), np.max(disp))
# %% 
# smallest dt is 0.03 but this is just the first step because of a small bug
print('minimum dt after first step ', np.min(dt[1:]))

# %% 
# show the displacment profile
ax = plt.gca()
plt.plot(np.cumsum(dt), disp)
ax.set_xlabel('time (s)')
ax.set_ylabel('displacement (microns)')
plt.show()
# %% 
def print_axis(arr):
    return '({:.3f}, {:.3f}, {:.3f})'.format(*arr)

# largest displacement is 0.27 which means velocity of 2.7 \mu m/s
# lets investigate those large displacements in detail
sort_idx = np.argsort(disp)
l_idx = sort_idx[-1]
print('largest displacement occurs at time {}s'.format( tr['time'][l_idx]) )
print('displacement {} -> {}'.format(xy[l_idx], xy[l_idx+1]))
axis = tr.get_frame()[1]
print('body axis {} -> {}'.format(print_axis(axis[l_idx]), print_axis(axis[l_idx+1])))
# lets decompose this displacement along the initial axis
thisdxy = dxy[l_idx]
primary_ax = axis[l_idx][:2]
FLOAT_TOL = 1e-6
assert(np.linalg.norm(primary_ax) - 1.0 < FLOAT_TOL)
primary_dx = np.dot(primary_ax, thisdxy) * primary_ax
print('this displacment', thisdxy)
sec_dx = thisdxy - primary_dx
print('after decomposing along the body axis:\n', primary_dx , sec_dx)


# %% [markdown]
# We find (surprising) that this large displacement is not due to a rotation
# so we need to rethink what can cause such a large displacement.
# In order to make progress we need a full output, not just 0.1s resolution data

# %% 
highresdir = "/home/dan/usb_twitching/run/c450a32/two_parameters_koch/pilivar_0050.00000_k_spawn_10.00000/testview/highres/"
highresdata = os.path.join(highresdir,"trackxy.dat")
with command.loadexplain('tracking data'):
    htr = readtrack.Track(highresdata)
with command.loadexplain('pili data'):
    with command.chdir(highresdir):
        ptdata = readtrack.piliset()[0]

# %%  
print('reorganise pili data by pilus')
pilusdata = pilush.reorganise(ptdata)
print('finished.')

# %% 
# first thing to do is find out if we can get the same large displacements as before
resolution = 0.1
htr.true_filter_to_resolution(resolution)
# since this is a new implementation of the resolution filtering method we make a short aside here to test it
assert(htr.slice[0] != htr.slice[1])
def test_filter(tr):
    xy, dxy, disp, dt = _displacement(htr)
    sort_idx = np.argsort(disp)
    l_idx = sort_idx[-1]
    print('largest displacement occurs at time {}s'.format( tr['time'][l_idx]) )
    print('displacement {} -> {}'.format(xy[l_idx], xy[l_idx+1]))
    return htr.slice[sort_idx]

print("Sanity check that after full output and setting resolution we recover the same result")
sort_idx = test_filter(htr)
# Sanity Check complete

# %%
# Examine the retraction events during this displacement
htr.clear_filter()
time = htr['time']
l_idx = sort_idx[-1]
l_idx1 = np.searchsorted(time, time[l_idx]+0.1)
print('there are {} MC steps between {:.3f} and {:.3f}'.format(
    l_idx1-l_idx, time[l_idx], time[l_idx1]))
print('npili is {:.3f}'.format( np.mean(htr['npili'][l_idx:l_idx1])))
nbound = np.mean(htr['nbound'][l_idx:l_idx1])
print('nbound is {}, (min,max) = ({},{})'.format(
    nbound, np.min(nbound), np.max(nbound)))
# so the binding number doesn't change
l_process = htr['process'][l_idx:l_idx1]
from collections import Counter
proc = Counter(l_process)
print('processes', proc)
# to examine the pili behvaiour we need load the event or pili data
# %%
# ptdata only includes bound pili (by default)
context, lpt = ptdata[l_idx]
start_time, end_time = time[l_idx], time[l_idx1]
print('context', context)
boundpidx = lpt['pidx']
print("so bound pili during large displacement are ", boundpidx)
for pidx in boundpidx:
    print('for pilus ', pidx)
    pdata = pilusdata[pidx]
    t = pdata['time']

    leq = pdata['pleq']; plength = pdata['plength']
    start_idx, end_idx = np.searchsorted(t, start_time),np.searchsorted(t,end_time)
    dl = leq[start_idx] - leq[end_idx]
    nret = int(dl//0.004)
    print('number of retractions {:d}, thats {:d}/s'.format(nret, int(nret/0.1)))
    bounddl = plength[start_idx] - plength[end_idx]
    print('length {:.4f} -> {:.4f}'.format(leq[start_idx], leq[end_idx]))
    print('leq change is {:.4f}, bound length change is {:.4f}'.format(dl, bounddl))
    print()
# so how is is that the body displaces by ~0.27?
# the retraction rate is set to ~187.5
# %%
# might want to pull out the retraction events from the tracking data 
# but tracking data seems to be missing pidx column ...

# %%
# lets plot the trajectory
ax = plt.gca()
x, y = htr['x'][l_idx:l_idx1], htr['y'][l_idx:l_idx1]
ax.plot(x, y)
ax.plot(x[0], y[0], marker='o')
ax.plot(x[-1], y[-1], marker='D')

ax.set_aspect('equal')
plt.show()
# displacement [21.02135177  1.7527593 ] -> [21.27734907  1.84484082]

# %%
# match up the displacements to retraction events
pidx = htr['pidx'][l_idx:l_idx1]
pdx = {}
header = ['time', 'ptime', 'x', 'y', 'px', 'py']
dtype = [(h, 'f4') for h in header]
for idx in boundpidx:
    trackidx = pidx == idx
    assert(trackidx[0] == False)
    # prev = np.nonzero(trackidx)[0]-1
    prev = np.roll(trackidx, shift=-1)
    track = htr[:][l_idx:l_idx1][trackidx]
    prevt = htr[:][l_idx:l_idx1][prev]
    x = track['x']; y = track['y'] 
    px = prevt['x']; py = prevt['y']
    time = track['time']; ptime = prevt['time']
    data = list(zip(*[time, ptime, x, y, px, py]))
    structure_array = np.array(data, dtype=dtype)
    pdx[idx] = structure_array

# %%
# 
ax = plt.gca()
alldisp = []
for idx in boundpidx:
    struct = pdx[idx]
    # displacements due to retraction events
    adx = struct['x'] - struct['px']
    ady = struct['y'] - struct['py']
    adisp = np.sqrt(adx**2 + ady**2)
    print(adisp)
    ax.axhline(0.004,c='k',linestyle='--')
    ax.plot(struct['time'], adisp, linestyle='none', marker='x', label=f'pidx {idx}')
    alldisp.append(adisp)
maxy = np.max(np.concatenate(alldisp))
ax.set_ylim(0.0,maxy)
ax.legend()
ax.set_ylabel
plt.show()
# so we see that we need to look at individual shrink events and 
# try to understand why we consistently get larger displacements 
# than the shrink distance

# %%
# frakenstein together the trajectory from these displacements
ax = plt.gca()
alldisp = []
color = iter(['b', 'r'])
markers = iter(['o','x'])
handles, labels = [], []
for idx in boundpidx:
    c = next(color)
    mark = next(markers)
    struct = pdx[idx]
    for row in struct:
        style = {'c':c, 'marker':mark}
        h, = ax.plot([row['px'],row['x']],[row['py'],row['y']], **style)
    handles.append(h)
    labels.append(str(idx))
    
ax.legend(handles, labels)
ax.set_aspect('equal')
plt.show()
# so particularly pilus 785 tends to have larger displacements as 
# time progresses 
# %%
# lets examine the situation of pilus 785 in more detail
thispidx = 785
pdata = pilusdata[thispidx]
datatime = pdata['time']
pstart_idx = np.searchsorted(datatime, start_time)
pend_idx = np.searchsorted(datatime, end_time)

l_frame = pilush._construct_frame(htr[:][l_idx])
def get_anchor(data, i):
    return base.Vector3d(data['anchor_x'][i],
            data['anchor_y'][i],
            data['anchor_z'][i]
        )

l_anchor = get_anchor(pdata, pstart_idx)
l_labanchor = l_frame.to_lab(l_anchor)
print('the lab anchor for pilus {} at the start of the action is:\n {}'.format(
    thispidx, l_labanchor
))
# its compelling to note that during this action we start close to
# the vertical configuration and reach it at the end of the 0.1s interval

# %%
# lets construct the triangle
R = l_labanchor.z # vertical
L = pdata['pleq'][pstart_idx] # hypotenuse
x = np.sqrt(L**2 - R**2)
def print_triangle(x, R, L):
    print('Triangle: {:.4f} {:.4f} {:.4f}'.format(x, R, L))
print_triangle(x, R, L)
# then after one shrink
L_ = L - 0.004
x_ = np.sqrt(L_**2 - R**2)
print_triangle(x_, R, L_)
dx = x - x_
print('so horizontal changed by', dx)
# The high velocity displacements is just simple geometry!
shrink = 0.004
def triangle_dx(H, L, shrink=shrink):
    x = np.sqrt(L**2 - H**2)
    x_ = np.sqrt((L-shrink)**2 - H**2)
    return x - x_

H = 0.5
Lspace = np.linspace(H+shrink, 3*H)
dx = [triangle_dx(H, l) for l in Lspace]
ax = plt.gca()
ax.axhline(shrink, c='k',  linestyle = '--')
ax.axvline(H, linestyle='--')
ax.set_ylim(0,0.03)
ax.set_ylabel('horizontal displacement for one retraction')
ax.set_xlabel('pili length')
ax.plot(Lspace, dx, marker='x')


# %%
etr = tr.extend_projected_axis_by_radius()

# %%
# plot the track side by side
# zoom in on a small part of the track
if verbose:
    fig, ax = plt.subplots()
    def plot_debug_tracks(tr, etr):
        # blue
        ax.plot(tr['x'], tr['y'], color='b')
        ax.plot(tr['trail_x'], tr['trail_y'], color='slateblue')
        # red
        ax.plot(etr['x'], etr['y'], color='red')
        ax.plot(etr['trail_x'], etr['trail_y'], color='salmon')
        ax.set_aspect('equal')

    # shorten
    tr_ = tr.copy()
    etr_ = etr.copy()
    N = 100
    tr_._track = tr_._track[:N]
    etr_._track = etr_._track[:N]
    plot_debug_tracks(tr_, etr_)
# plot_debug_tracks(tr, etr)
# %%
# now check extrema
fig, axes = plt.subplots(2,1,sharex=True,sharey=True)
ax1, ax2  = axes
def plot_disp(ax, tr):
    print('etr')
    xy, dxy, disp, dt = _displacement(tr)
    print(twutils.pretty_describe(disp))
    plt.plot(np.cumsum(dt), disp)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('displacement (microns)')

print('tr')
print(twutils.pretty_describe(disp))
ax1.plot(np.cumsum(dt), disp)
plot_disp(ax2, etr)
plt.show()
# rotations should mean a larger average displacments for the processed data
# although the effect is small in this simulation

# curiously the maximum displacement of etr is lower. is this a bug(?)
# when we read experimental data should we automatically apply this transformation?

# %% [markdown]
# after this analysis we discovered that the extreme velocities were 
# are a result of high anchor flexibility and constant retraction speed
# and not a bug
