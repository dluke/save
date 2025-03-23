
import os
import sys
join = os.path.join
import mdl
import annealing
import matplotlib.pyplot as plt

import _fj
import json

args = sys.argv[1:]
if (len(args) != 1):
    print("provide a folder path")
    sys.exit()

target = args[0]

with open(join(target, 'config.json'), 'r') as f:
    config = json.load(f)

trackidx =  config['trackidx']
track = _fj.trackload_original([trackidx])[0]
data = mdl.get_lptrack(track)

solver = annealing.Solver.load_state(path=join(target, "solver"))

fs = 200
fig, ax = plt.subplots(figsize=(fs,fs))
annealing.model_plot(ax, solver.anneal, data)
path = join(target, "final_state_f200.svg")
print("writing to ", path)
plt.savefig(path)


