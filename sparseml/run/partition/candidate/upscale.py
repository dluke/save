
import os
import sys
join = os.path.join
import pickle
import mdl
import pwlpartition
import matplotlib.pyplot as plt

import _fj
import json

def load_pkl(at):
    if not at.endswith('.pkl'): at += '.pkl'
    with open(at, 'rb') as f: return pickle.load(f)


args = sys.argv[1:]
if (len(args) != 1):
    print("provide a folder path")
    sys.exit()

target = args[0]


with open(join(target, 'data.pkl'), 'rb') as f:
    data = pickle.load(f)

solver = pwlpartition.Solver.load_state(path=join(target, "solver"))
initial = load_pkl(join(target, "initial_guess"))

max_fs = 200
size = data.get_bounding_size()
fs = min(6.0 * max(size), max_fs)

print('use figsize', fs)
pwlpartition.model_plot(solver, data, fs=fs)
path = join(target, "final_state_upscaled.svg")
print("writing to ", path)
plt.savefig(path)

fig, ax = plt.subplots(figsize=(fs,fs))
pwlpartition.simple_model_plot(ax, initial, data)
path = join(target, "initial_state_upscaled.svg")
print("writing to ", path)
plt.savefig(path)


