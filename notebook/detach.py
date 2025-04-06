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
# Model Attachment and Detachment are highly simplified processes. A pilus attaches if its current chain configuration intersects the surface. The attachment check occurs automatically after the body or pilus is updated.
# In order to implement the surface dwell timescale $\tau_\mathrm{dwell}$ (Tala et al.), detachment is checked on this timescale. However, because attachment occurs automatically on touching the surface, the pilus must
# clear the surface before it can detach. This is done by coupling detachment with a retraction event which shrinks the pilus by a tiny amount. Long pili may also clear the surface by resampling their chain.
# Usually however, the pili need to shrink and become taut before they can detach. This additional constraint on the pilus detachment means that generally $\tau_\mathrm{dwell} \neq \tau_\mathrm{bound}$.
# $\tau_\mathrm{dwell}$ is the model pararmeter, while $\tau_\mathrm{bound}$ is what is observed in experiment (Tala et al.). This discrepancy is undesireable and we would like it to be minimised.
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import twutils
import parameters
import plotutils
import command
import twanalyse
import pilush
import readtrack
# %%

verbose = False

# %%
notename = 'detach'
notedir = os.getcwd()
# simdir = os.path.join(notedir, "exampledata/two_parameters/pilivar_0013.00000_k_spawn_05.00000/")

# single track
simdir = "/home/dan/usb_twitching/run/two_parameter_model/two_parameters_talaI/single"

# debugging
simdir = "/home/dan/usb_twitching/debug_record/two_parameter_model/two_parameters_talaI/single_19c4b81"

simdir = "/home/dan/usb_twitching/debug_record/two_parameter_model/two_parameters_talaI/single_3433301/withvtk"

simdir = "/home/dan/usb_twitching/debug_record/two_parameter_model/two_parameters_talaI/single_3433301/long"

# set min_length to an appropriate value
simdir = "/home/dan/usb_twitching/debug_record/two_parameter_model/two_parameters_talaI/single_3433301/long/min_length"

simdir = "/home/dan/usb_twitching/debug_record/two_parameter_model/two_parameters_talaI/single_8ac9944/long/min_length" 

simdata = os.path.join(simdir, "data/")
print('data at: ', simdata)


# %%
with command.chdir(simdir):
    print('reading dataset')
    evdataset = readtrack.eventset()
    trs = readtrack.trackset()
    ptrs  = readtrack.piliset()
    args = parameters.thisread()

evdata = evdataset[0]
tr = trs[0]
ptr = ptrs[0]

# %%
print('reorganising pilus data...')
pilusdata = pilush.reorganise(ptrs[0])
print('done.')

# %%

import eventanalyse
with command.chdir(simdir):
    args = parameters.thisread()
    sdata = eventanalyse.lifetime()
    # summary = twanalyse.summary()
# %%
print('spawned', np.count_nonzero(evdata['trigger'] == 'spawn'))
print('dissolved', np.count_nonzero(evdata['trigger'] == 'dissolve'))
print()
print('event fields', evdata.get_dtype().names)
print()
print('track fields', tr.get_dtype().names)

# %%
# want to know the number of failed detachments
# eventanalyse.failed_release
# its a lot
# want to know if these release events are happening with retraction motor
# as antipicated
i_release = evdata['process'] == 'release'
print('number of release events with ret motor', np.count_nonzero(evdata['ret_motor'][i_release]))
print('number of release events with ext motor', np.count_nonzero(evdata['ext_motor'][i_release]))
# yes they are

# is pilus taut?
# taut if leq < |attach - anchor|
i_norelease = evdata['trigger'] == 'no_release'
taut_fraction = np.count_nonzero(evdata['istaut'][i_release])/np.count_nonzero(i_release)
f_taut_fraction = np.count_nonzero(evdata['istaut'][i_norelease])/np.count_nonzero(i_norelease)
norelease_fraction = np.count_nonzero(i_norelease)/np.count_nonzero(i_release)
print('fraction of release events that occur while taut', taut_fraction)
print('fraction of failed release that occur while taut', f_taut_fraction)
print('failed release fraction', norelease_fraction)

# %%
# reconstruct the geometry ...
# are the pili nearly vertical?
e_z = np.array([0,0,1])
countidx = 0
i_candidate = np.logical_and(i_norelease, evdata['istaut'])
cand_pidx = evdata[i_candidate][countidx]['pidx']
cand_ev = evdata[evdata['pidx'] == cand_pidx]
print('candidate pilus', cand_pidx)
ptime = pilusdata[cand_pidx]['time']
parb = pilusdata[cand_pidx]
print(ptime[0],ptime[-1])
print(ptime.size)
cand_norelease = evdata[np.logical_and(i_norelease, evdata['pidx'] == cand_pidx)]

# %% 
# breaking down pilush.get_segment ...
tstart, tend = ptime[0], ptime[-1]
stidx = np.argwhere(tr['time'] == tstart)[0][0]
seidx = np.argwhere(tr['time'] == tend)[0][0]
trwhole = tr[stidx:seidx+1]
i_truerelease = np.logical_and(
    cand_ev['process'] == 'release', 
    cand_ev['trigger'] == 'release')

release_times = cand_ev['time'][i_truerelease]
release_time = release_times[0]
release_idx = np.searchsorted(trwhole['time'], release_time, side='left')
trseg = trwhole[:release_idx]

# %% 
print(stidx, seidx)
cand_events = evdata[evdata['pidx'] == cand_pidx]
for ev in cand_events:
    print(ev['time'], ev['process'], ev['trigger'])

Nc = len(range(stidx, seidx+1))
# print(Nc, ptime.size)
if len(range(stidx, seidx+1)) != ptime.size:
    print('warning: pilus data not contiguous')

# %%
# plot a pilus length
ax = plt.gca()
ax.plot(ptime, parb['pleq'], alpha=0.5, label='leq')
ax.plot(ptime, parb['plength'], alpha=0.5, label=r'$|anchor - attach|$')
for time in cand_norelease['time']:
    ax.axvline(time, linestyle='--', color='k', alpha=0.2)
for time in release_times:
    ax.axvline(time, linestyle='-', color='r', alpha=0.8)
ax.set_xlabel('time (s)')
ax.set_ylabel(r'length $\mu m$')
ax.legend()
plt.show()

# %%
# reconstruct geometry
trseg, prowdata = pilush.get_segment(tr, ptr, cand_ev, cand_pidx, ptime)
frame = [pilush._construct_frame(trdata) for trdata in trseg]
print(trseg.size, len(prowdata))
print(prowdata[0].dtype.names)
    
# %%
print('constructing lab axes...')
labanchor = [pilush._construct_anchor(f, row) for f, row in zip(frame, prowdata)]
labaxis = [pilush._construct_axis(a, row) for a, row in zip(labanchor, prowdata)]
print('done.')
# %%
# want to know if distance to intersection of free pilus is less
# than pleq - detach_grace_length
# first no release

# eventdata
fnr = cand_norelease[0]


norelease_idx = np.searchsorted(trseg['time'], fnr['time'], side='left')
paxis = labaxis[norelease_idx]
anch = labanchor[norelease_idx]
print('lab axis', paxis)
print('anchor', anch)
a = -anch[2]/paxis[2]
print('free pilus intersection distance ', a)
print('bound pilus length', fnr['plength'])
print('free pilus length', fnr['pleq'])
# and in fact the reconstructed pilus intersection length is larger ...

trackrow = trseg[norelease_idx]

verbose = False
if verbose:
    for name in fnr.dtype.names:
        print(name, fnr[name])
    print()
    for name in trackrow.dtype.names:
        print(name, trackrow[name])
import pili
import test_detach
test_detach.create_test_data(trackrow,fnr,args,0)


# %%
e_z = np.array([0,0,1])
angle = [np.dot(e_z, labax) for labax in labaxis]
ax = plt.gca()
ax.plot(trseg['time'], angle)
ax.set_ylim([-1.1,0])
ax.set_ylabel(r'$\vec{a} \cdot e_z$')
plt.show()

# %%
# check head and tail z
ax= plt.gca()
ax.plot(trseg['time'], trseg['z'] ,label='z')
ax.plot(trseg['time'], trseg['trail_z'], label='trail z')
ax.legend()

plt.show()

# %%
ax = plt.gca()
ax.plot(tr['time'], tr['npili'], label=r'$N_{\textrm{pili}}$')
ax.plot(tr['time'], tr['nbound'], label=r'$N_{\textrm{bound}}$')
ax.plot(tr['time'], tr['ntaut'], label=r'$N_{\textrm{taut}}$')
ax.set_xlabel('time (s)')
ax.legend()
ax

# %%
v = tr.get_head_v()
speed = np.linalg.norm(v, axis=1)
print(np.mean(speed[:500]))
print(np.mean(speed))
idx = speed == np.max(speed)
maxspeedtime = tr['time'][1:]
plot_speed = speed.copy()
plot_speed[speed > np.quantile(speed, 0.99)] = np.nan
print(maxspeedtime, np.max(speed))
print(tr[1:][idx])

ax = plt.gca()
ax.set_title('speed profile')
ax.plot(tr['time'][1:], plot_speed)
ax
