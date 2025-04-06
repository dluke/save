# %% [markdown]
# compute lengths of the leading and trail pole trajectories for walking set

# %% 
import os
import sys
import json
join = lambda *x: os.path.abspath(os.path.join(*x))
import numpy as np
norm = np.linalg.norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import shapeplot
import filesystem

import _fj
import plotutils
import command
import twanalyse
import fjanalysis
import twanimation
import pili
import stats

import pandas as pd
import seaborn as sns
import pili.publication as pub

import pili.support as support
from pili.support import make_get, get_array

from tqdm import tqdm

# %% 

cache_dir = "plos_classification/"
walking_idx = np.loadtxt(join(cache_dir, "new_walking.npy")).astype(int)
walking_idx
tracks = _fj.lintrackload(walking_idx)

# %% 

idx = 0
track = tracks[idx]
# lead = track.get_head2d()
# trail = track.get_trail2d()


# %%
leaddx = track.get_step_displacement()
traildx = track.get_step_displacement(trail=True)
duration = track.get_duration()
lead_displacement = np.sum(leaddx)/duration
trail_displacement = np.sum(traildx)/duration
lead_displacement, trail_displacement

# %%
kmsd = [twanalyse.kmsd(tr) for tr in tracks]
kmsd_trail = [twanalyse.kmsd(tr, trail=True) for tr in tracks]

# %%
# sns.histplot(kmsd)
sns.histplot(kmsd_trail)

# %%

lead_speed = []
trail_speed = []
for track in tracks:

	duration = track.get_duration()
	lead_speed.append( track.get_step_displacement().sum()/duration )
	trail_speed.append( track.get_step_displacement(trail=True).sum()/duration )


np.mean(lead_speed)
np.mean(trail_speed)

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(5,5))
	sns.scatterplot(lead_speed, trail_speed, ax=ax)
	line = np.linspace(0, np.max(lead_speed), 100)
	ax.plot(line, line, linestyle='--', c='orange', lw=2) 
	ax.set_aspect("equal")
	ax.set_xlabel("leading pole velocity (µm/s)")
	ax.set_ylabel("trailing pole velocity (µm/s)")
	

fig.tight_layout()
plt.savefig(join('brownian_walking', 'lead_trail_velocity.png'))

# %%

with mpl.rc_context({'font.size': 16}):
	fig, ax = plt.subplots(figsize=(5,5))
	sns.scatterplot(kmsd, kmsd_trail, ax=ax)
	line = np.linspace(0, 2, 100)
	ax.plot(line, line, linestyle='--', c='orange', lw=2) 
	ax.set_aspect("equal")
	ax.set_xlabel("leading KMSD")
	ax.set_ylabel("trailing KMSD")
	ax.xaxis.set_major_locator(plt.MaxNLocator(5))
	ax.yaxis.set_major_locator(plt.MaxNLocator(5))

fig.tight_layout()
plt.savefig(join('brownian_walking', 'lead_trail_kmsd.png'))

print(np.median(kmsd)), print(np.median(kmsd_trail))
