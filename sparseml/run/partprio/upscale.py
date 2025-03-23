
import os
import sys
join = os.path.join
import pickle
import mdl
import pwlpartition
import matplotlib.pyplot as plt

import _fj
import json

args = sys.argv[1:]
if (len(args) != 1):
    print("provide a folder path")
    sys.exit()

target = args[0]

# with open(join(target, 'config.json'), 'r') as f:
#     config = json.load(f)

# trackidx =  config['trackidx']
# track = _fj.trackload_original([trackidx])[0]
with open(join(target, 'data.pkl'), 'rb') as f:
    data = pickle.load(f)

solver = pwlpartition.Solver.load_state(path=join(target, "solver"))

max_fs 200
size = data.get_bounding_size()
fs = min(6.0 * max(size), max_fs)

print('use figsize', fs)
pwlpartition.model_plot(solver, data, fs=fs)
path = join(target, "final_state_upscaled.svg")
print("writing to ", path)
plt.savefig(path)


