

# information theory utilities
import numpy as np
import time


# https://en.wikipedia.org/wiki/Sample_entropy
def sampen(L, m, r):
    N = len(L)
    B = 0.0
    A = 0.0
    
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)



# %%
# utilities
class Timer:
    def __init__(self, message=''):
        self.message = message
        self.start = None
        self.end = None
        self.time = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end = time.perf_counter()
        self.time = self.end-self.start
        print(self.message, "timed {:.8f}s".format(self.time))
