import numpy as np
d = np.loadtxt('power_spectrum.dat')
k, P = d[:,0], d[:,1]   # k is k/k_star
mask = (k > 0.1) & (k < 10)
lnk = np.log(k[mask])
lnP = np.log(P[mask])
slope, intercept = np.polyfit(lnk, lnP, 1)
print(f"n_s = {1 + slope:.4f}")
