
# %% [markdown]
# explore the ruptures library

# %%

import matplotlib.pyplot as plt
import ruptures as rpt

# %% # generate signal
n_samples, dim, sigma = 200, 1, 4
n_bkps = 4  # number of breakpoints
signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma, delta=(8,12))

# detection
model = 'l2'
# algo = rpt.BottomUp(model=model, jump=1).fit(signal)
algo = rpt.Binseg(model=model, jump=1).fit(signal)
threshold = 1.0 * n_samples * sigma**2
print('threshold', threshold)
result = algo.predict(epsilon=threshold)


# algo = rpt.Pelt(model="rbf").fit(signal)
# result = algo.predict(pen=1)

print('result', result)

# display
rpt.display(signal, bkps, result)
plt.show()

# %%
c = rpt.costs.CostL2().fit(signal)
c.error(0,100)
# 10 * sigma ** 2
