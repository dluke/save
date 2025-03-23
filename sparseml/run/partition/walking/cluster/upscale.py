
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

fs = 20
# pwlpartition.model_plot(solver, data, fs=fs)
fig, ax = plt.subplots(figsize=(fs,fs))
pwlpartition.simple_model_plot(ax, solver.partition.model, data)
path = join(target, "final_state_upscaled.png")
print("writing to ", path)
plt.savefig(path)
plt.show()


