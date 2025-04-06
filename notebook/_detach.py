
# %%
print('pili data', ptr[0][1].dtype.names)
max_cycles = np.array([np.max(ptrdat[1]['cycles'], initial=0) for ptrdat in ptr])
violate_max_cycles = [i for i ,mc in enumerate(max_cycles) if mc > 1]
print(max_cycles[max_cycles>1])
print(violate_max_cycles)
varg1, varg2 = violate_max_cycles[:2]

# %%
from tabulate import tabulate
def to_table(arr):
    return tabulate(arr, arr.dtype.names)

print(to_table(ptr[varg1][1]))
# print(to_table(ptr[vargend1][1]))
# pilus 3450 has 17 cycles

# %%
cand_idx = 3450
cand_idx = 2099
candidate = pilusdata[cand_idx]
print(candidate.keys())
print(candidate['time'])
# different pili given the same pidx
# %%
ax = plt.gca()
def plot_candidate(ax, candidate):
    time = candidate['time']
    length = candidate['pleq']
    clength = candidate['plength']
    ax.plot(time, length, label='length', marker='o')
    ax.plot(time, clength, label='contour length', marker='x')
    ax.legend()
    return ax
plot_candidate(ax, candidate)

# %%
# compare with a random pilus
rg = np.random.default_rng(1)
rpidx = rg.choice(np.array(list(pilusdata.keys())))
rpidx = list(pilusdata.keys())[0]
print('pidx', rpidx)
print('size ', pilusdata[rpidx]['pidx'].size)
plot_candidate(plt.gca(), pilusdata[rpidx])
# our 0.1s time resolution is pretty much insufficient


