
# shift code out of sampen.py notebook

# %% 
# what about using the linearisation coarse graining,
# keep in mind typical step size should be >> the time resolution(?)
dstepbasis = [0.01, 0.02, 0.03, 0.04, 0.06, 0.09, 0.12, 0.18, 0.24]
# we cannot just replace the granular function because the linearisation operation 
# is defined on the 2d trajectory

def linearise(x2data, dstep):
    # x2data.shape = (N, 2)
    x, y = x2data.T
    N = len(x)
    step_idx = [0]
    step_displacement = []
    i = 0
    i_f = 1
    while i_f < N:
        d = np.sqrt( (x[i]-x[i_f])**2 + (y[i]-y[i_f])**2 )
        if d > dstep:
            step_idx.append(i_f)
            step_displacement.append(d)
            i = i_f
            i_f = i+1
        else:
            i_f += 1
    # the last partial step should be thrown away
    return np.array(step_idx), np.array(step_displacement)

def multiscale_sampen2d(v2data, m, r):
    x2data = np.cumsum(v2data, axis=0)

    lengthscale = dstepbasis

    _entropy = []
    for dstep in lengthscale:
        _, disp = linearise(x2data, dstep)
        print("linearising at dstep = {} gives N = {}".format(dstep, len(disp)))
        # course_data = disp/tiem
        coarse_data = disp
        entropy = sampen(coarse_data, m, r)
        _entropy.append(entropy)
    return lengthscale, _entropy

x2data = t0velocity[:N]
basis, entropy = multiscale_sampen2d(x2data, m, r)

# %% 
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(basis, entropy, marker='o')
ax.set_ylabel("sample entropy")
ax.set_xlabel(r"$\delta_{\mathrm{step}}$")
