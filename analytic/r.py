import numpy as np
bg = np.loadtxt('background.dat', comments='#')
N, phi, dphi, eps, logH = bg[:,0], bg[:,1], bg[:,2], bg[:,3], bg[:,4]
N_tot = N[-1]
N_star = N_tot - 60.0
i = np.argmin(np.abs(N - N_star))
eps_star = eps[i]
r_sr = 16 * eps_star
print(f"epsilon_H* = {eps_star:.4e}")
print(f"r (slow-roll) = {r_sr:.4f}")
