
import os
import sys
join = os.path.join
import pickle
import mdl
import pwlpartition
import matplotlib.pyplot as plt

import _fj
import json

import argparse

args = sys.argv[1:]
if (len(args) != 1):
    print("using current directory")
    target = '.'
else:
    target = args[0]

with open(join(target, 'data.pkl'), 'rb') as f:
    data = pickle.load(f)

with open(join(target, 'truth.pkl'), 'rb') as f:
    pwl = pickle.load(f)



solver = pwlpartition.Solver.load_state(path=join(target, "solver"))



sx, sy = data.cut(0,10).get_bounding_size()
lx, ly = data.cut(0,10).get_limits(buffer=(0.05*sx, 0.5*sy))

size = data.get_bounding_size()
# max_fs = 200
# fs = min(6.0 * max(), max_fs)
fs = 40
print('using figure size', fs)

print('use figsize', fs)
fig, ax = plt.subplots(figsize=(fs,fs))
is_outlier = solver.get_outliers()
_config = {'h_outlier': True, 'r' : solver.r}

# mdl.plot_model_on_data(ax, solver.partition.model, data, intermediate={'is_outlier':is_outlier}, config=_config)

truth_style = {"linestyle": '--', 'lw':2, 'alpha':0.5, 'label':'truth'}
model_style = {"linestyle": '-', 'lw':4, 'alpha':0.5, 'label':'model'}
ptlkw = {"linestyle":'--', "marker":"o", "alpha":0.3}

def local_plot_model(ax, model, style={}):
    ax.plot(model.x, model.y, **style)
    ax.set_aspect('equal')

local_plot_model(ax, pwl, truth_style)
local_plot_model(ax, solver.partition.model, model_style)
ax.plot(data.x, data.y, label='data', **ptlkw)

# handles = ax.get_legend().legendHandles
# labels = [t.get_text() for t in ax.get_legend().texts]
ax.legend(fontsize=20, loc=(1.04, 0))

outname = "final_state.svg"
path = join(target, outname)
print('writing to', path)
plt.tight_layout()
plt.savefig(path, bbox_inches="tight")


ax.set_xlim(lx)
ax.set_ylim(ly)
outname = "final_state_cropped.png"
path = join(target, outname)
print("writing to ", path)

plt.tight_layout()
plt.savefig(path, bbox_inches="tight")


