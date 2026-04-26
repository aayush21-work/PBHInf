#!/usr/bin/env python3
"""
Slow-roll inflationary observables for the potential

    V(phi) = V0 * phi^2 / (1 + phi)^2

Treats phi as a canonically normalized single scalar field (M_Pl = 1).

Computes:
  - Field value at end of inflation, phi_end, from eps_V = 1
  - Field value at horizon crossing, phi_*, from the N_e integral
  - Slow-roll parameters (eps_V, eta_V) at phi_*
  - Observables: n_s = 1 - 6 eps_V + 2 eta_V,   r = 16 eps_V
  - Amplitude normalization V0 from A_s = 2.1e-9

Usage:
    python slow_roll_ns_r.py              # default: N_e in {50, 55, 60}
    python slow_roll_ns_r.py --Ne 58      # single value
    python slow_roll_ns_r.py --scan       # scan Ne from 40 to 70
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import argparse


# Potential and derivatives 

def V(phi, V0=1.0):
    return V0 * phi**2 / (1.0 + phi)**2

def dV(phi, V0=1.0):
    return V0 * 2.0 * phi / (1.0 + phi)**3

def d2V(phi, V0=1.0):
    return V0 * 2.0 * (1.0 - 2.0 * phi) / (1.0 + phi)**4


#Slow-roll parameters 

def eps_V(phi):
    """epsilon_V = 1/2 (V'/V)^2"""
    dlogV = dV(phi) / V(phi)
    return 0.5 * dlogV**2

def eta_V(phi):
    """eta_V = V''/V"""
    return d2V(phi) / V(phi)


#End of inflation 

def find_phi_end():
    return 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * np.sqrt(2.0)))


# E-fold integral 

def N_of_phi(phi_star, phi_end):
    def f(p):
        return p**2 / 4.0 + p**3 / 6.0

    return f(phi_star) - f(phi_end)

def N_of_phi_numerical(phi_star, phi_end):
    integrand = lambda p: V(p) / dV(p)  
    val, _ = quad(integrand, phi_end, phi_star)
    return val


def find_phi_star(Ne_target, phi_end):
    f = lambda p: N_of_phi(p, phi_end) - Ne_target
    return brentq(f, phi_end + 1e-3, 50.0)


# Observables

def observables(phi_star):
    e = eps_V(phi_star)
    h = eta_V(phi_star)
    n_s = 1.0 - 6.0 * e + 2.0 * h
    r   = 16.0 * e
    return e, h, n_s, r


def compute_V0(phi_star, A_s=2.1e-9):
    """
    Normalize V0 from the scalar amplitude:
        A_s = V / (24 pi^2 eps_V) at phi_star (in M_Pl = 1 units)
        =>   V0 = A_s * 24 pi^2 eps_V * (1+phi_star)^2 / phi_star^2
    """
    e = eps_V(phi_star)
    u_star = phi_star**2 / (1.0 + phi_star)**2   # = V(phi_star) / V0
    return A_s * 24.0 * np.pi**2 * e / u_star


# ---------- Main report -----------------------------------------------------

def report(Ne_values):
    phi_end = find_phi_end()
    print(f"Potential: V(phi) = V0 * phi^2 / (1+phi)^2   [M_Pl = 1]")
    print(f"End of inflation: phi_end = {phi_end:.4f}  (eps_V = 1)")
    print()
    print(f"{'N_e':>5}  {'phi_*':>8}  {'eps_V':>11}  {'eta_V':>11}  "
          f"{'n_s':>8}  {'r':>8}  {'V0 [M_P^4]':>12}")
    print("-" * 78)

    for Ne in Ne_values:
        phi_star = find_phi_star(Ne, phi_end)
        e, h, n_s, r = observables(phi_star)
        V0 = compute_V0(phi_star)
        print(f"{Ne:>5}  {phi_star:>8.4f}  {e:>11.4e}  {h:>+11.4e}  "
              f"{n_s:>8.4f}  {r:>8.4f}  {V0:>12.4e}")

    print()
    print("Asymptotic (large N_e):")
    print("  n_s - 1 ~  -4/(3 N_e)  + O(N_e^{-4/3})")
    print("  r       ~  32/(6 N_e)^{4/3}")


def scan():
    phi_end = find_phi_end()
    import sys
    try:
        import matplotlib.pyplot as plt
        have_mpl = True
    except ImportError:
        have_mpl = False

    Ne_array = np.arange(40, 71, 1)
    ns_array, r_array = [], []
    for Ne in Ne_array:
        phi_star = find_phi_star(Ne, phi_end)
        _, _, n_s, r = observables(phi_star)
        ns_array.append(n_s)
        r_array.append(r)

    data = np.column_stack([Ne_array, ns_array, r_array])
    np.savetxt('ns_r_scan.dat', data,
               header='N_e  n_s  r', fmt='%8.4f')
    print("Wrote ns_r_scan.dat (Ne, n_s, r) for N_e in [40, 70]")

    if have_mpl:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(Ne_array, ns_array, 'b-')
        ax1.axhspan(0.9743 - 0.0034, 0.9743 + 0.0034, alpha=0.2, color='purple',
                    label='P-ACT-LB 1σ')
        ax1.set_xlabel(r'$N_e$')
        ax1.set_ylabel(r'$n_s$')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(Ne_array, r_array, 'b-')
        ax2.axhline(0.036, ls='--', color='red', label='BK18 2σ')
        ax2.set_xlabel(r'$N_e$')
        ax2.set_ylabel(r'$r$')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('ns_r_scan.png', dpi=120)
        print("Wrote ns_r_scan.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ne', type=float, default=None,
                        help='single N_e value')
    parser.add_argument('--scan', action='store_true',
                        help='scan Ne from 40 to 70, save data and plot')
    args = parser.parse_args()

    if args.scan:
        scan()
    elif args.Ne is not None:
        report([args.Ne])
    else:
        report([50, 55, 60])


if __name__ == '__main__':
    main()
