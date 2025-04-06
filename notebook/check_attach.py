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
# We can classify attachment events by event that precedes or causes them: extension, resampling, mdstep and release, note that:  
# - since we now check that pili can release before allowing the event to have positive probability this should not happen
# - mdstep attachment events are due the body position updating, since this happens incrementally we expect 
# the inter section lengths to be on the order of ~d_bound = 0.004 or a small factor of this for larger body movements
# caused by pilus release or body rotations.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import wrinterval as wr
import parameters
import readtrack
import plotutils
import command
import eventanalyse
# %%
notename = 'check_attach'
notedir = os.getcwd()

simdir = os.path.join(notedir, "exampledata/two_parameters/pilivar_0013.00000_k_spawn_05.00000/")

# debugging
simdir = "/home/dan/usb_twitching/debug_record/two_parameter_model/two_parameters_talaI/single_19c4b81"
simdata = os.path.join(simdir, "data/")

# %%
# load
evtrs = readtrack.eventset(simdata)

# %%
def _analyse(evtr):
    at_type = eventanalyse._classify(evtr)
    # 
    dd = {}
    dd['at_type'] = at_type
    dd['at_count'] = { k: len(v) for k, v in at_type.items()}
    dd['at_inter'] = { k : evtr['inter_excess'][idx] for k, idx in at_type.items() }
    dd['at_excess'] = { k : evtr['pleq'][idx] - evtr['plength'][idx] for k, idx in at_type.items() }
    return dd
dds = [_analyse(evtr) for evtr in evtrs]
keys = dds[0]['at_count'].keys()
at_count = {k: np.sum([dd['at_count'][k] for dd in dds]) for k in keys}
at_inter_m = {k: np.mean([np.mean(dd['at_inter'][k]) for dd in dds]) for k in keys}
at_excess_m = {k: np.mean([np.mean(dd['at_excess'][k]) for dd in dds]) for k in keys}

# %%
print(at_count)
# %% [markdown]
# Looking at the counts of each attachment trigger, extension is the most common followed by resampling which
# is expected and intended. An tiny insignificant portion of attachments are still classified as caused by release
# even though I attempted to make this impossible.
# A small but significant number of attachments are caused by updating the cell body. This is worth a closer look.
# %%
plt.style.use(plotutils.get_style('bigax')) 
# not the most appropriate style

# %%
# plot a bar graph using counts dictionary
fig = plt.figure(figsize=(16,10))
ax = fig.gca()
ax.set_ylabel('Number of attachments')
ax.set_xlabel('Cause of attachment')
ax.set_yticks([])
fig._plot_name = 'attach_number'
plotutils.plot_bar(ax, at_count)
fig.tight_layout()
plt.show()
# %%
#
# also plot average intersection length of attachments
fig = plt.figure(figsize=(16,10))
ax = fig.gca()
ax.set_ylabel('Avg. Intersecton length ($\mu m$)')
ax.set_xlabel('Cause of attachment')
ax.locator_params(nbins=4)
plt.yticks(fontsize=30)
fig._plot_name = 'inter_length'
plotutils.plot_bar(ax, at_inter_m)
fig.tight_layout()
plt.show()

# %%
# also plot average excess length of attachments
fig = plt.figure(figsize=(16,10))
ax = fig.gca()
ax.set_ylabel('Avg. Excess Length ($\mu m$)')
ax.set_xlabel('Cause of attachment')
ax.locator_params(nbins=4)
plt.yticks(fontsize=30)
fig._plot_name = 'excess_length'
plotutils.plot_bar(ax, at_excess_m)
fig.tight_layout()
plt.show()

# %% [markdown]
# As expected attachments caused by resampling dominate the excess length, however,
# mdstep has a significant count and large values of excess length despite the fact that incremental updates
# to the cell position should cause reletivly small intersections except in the case of rotations of the body.
# Lets extract the event history of these pili to look for clues.

# %% 
def _get_history(evtr, at_type):
    h = evtr[:][at_type['mdstep']]
    # sort into pili?
    return h
    # history = {pidx: evtr[:][evtr['pidx'] == pidx] for pidx in md_pidx}
h0 = _get_history(evtrs[0], dds[0]['at_type'])

# %%
# re-create the writer
event = wr.Track3d.setup_event_3d('event.tmp')
s = ''.join([event.compile_line(row) for row in h0])
with open('tmp.dat', 'w') as ftmp:
    ftmp.write(s)

# %%
