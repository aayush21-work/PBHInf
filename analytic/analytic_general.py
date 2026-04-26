#!/usr/bin/env python3
"""
Slow-roll inflationary observables for the family of potentials

    V(phi) = V0 * phi^n / (1 + phi)^n      n = 2, 3, 4, 5, 6, 7

M_Pl = 1 throughout.

For general n:
    V'/V  =  n / [phi (1+phi)]
    V''/V =  n [n(1-phi) - 2 phi] / [phi^2 (1+phi)^2]

phi_end is found numerically from eps_V(phi_end) = 1.
phi_star is found numerically from the e-fold integral.

Usage:
    python analytic_n.py                   # table for n=2..7, Ne in {50,55,60}
    python analytic_n.py --Ne 58           # single Ne
    python analytic_n.py --scan            # ns-r plot for each n
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import argparse


# ---------- Potential family -------------------------------------------------

def eps_V(phi, n):
    """epsilon_V = 1/2 (V'/V)^2  =  n^2 / [2 phi^2 (1+phi)^2]"""
    return 0.5 * (n / (phi * (1.0 + phi)))**2

def eta_V(phi, n):
    """eta_V = V''/V  =  n(n-1-2phi) / [phi^2 (1+phi)^2]
    Derived from d/dphi [V'/V] + (V'/V)^2:
      = -n(1+2phi)/[phi^2(1+phi)^2] + n^2/[phi^2(1+phi)^2]
      = n(n-1-2phi) / [phi^2(1+phi)^2]
    Reduces correctly to 2(1-2phi)/[phi^2(1+phi)^2] at n=2.
    """
    return n * (n - 1.0 - 2.0 * phi) / (phi * (1.0 + phi))**2


# ---------- End of inflation -------------------------------------------------

def find_phi_end(n):
    """Solve eps_V(phi_end) = 1  =>  phi_end*(1+phi_end) = n/sqrt(2)"""
    # phi^2 + phi - n/sqrt(2) = 0
    discriminant = 1.0 + 4.0 * n / np.sqrt(2.0)
    return 0.5 * (-1.0 + np.sqrt(discriminant))


# ---------- E-fold integral (numerical for all n) ---------------------------

def N_of_phi_numerical(phi_star, phi_end, n):
    """N = integral_{phi_end}^{phi_star}  V/V' dphi
             = integral  phi(1+phi)/n  dphi
             = [phi^2/2 + phi^3/3]_{phi_end}^{phi_star} / n
    """
    # Actually analytic for any n — V/V' = phi(1+phi)/n
    def antideriv(p):
        return p**2 / 2.0 + p**3 / 3.0
    return (antideriv(phi_star) - antideriv(phi_end)) / n


def find_phi_star(Ne_target, phi_end, n):
    f = lambda p: N_of_phi_numerical(p, phi_end, n) - Ne_target
    return brentq(f, phi_end + 1e-4, 200.0)


# ---------- Observables & normalization -------------------------------------

def observables(phi_star, n):
    e = eps_V(phi_star, n)
    h = eta_V(phi_star, n)
    n_s = 1.0 - 6.0 * e + 2.0 * h
    r   = 16.0 * e
    return e, h, n_s, r

def compute_V0(phi_star, n, A_s=2.1e-9):
    """A_s = V / (24 pi^2 eps_V)  =>  V0 = A_s * 24 pi^2 eps_V / u_star
       where u_star = (phi/(1+phi))^n evaluated at phi_star."""
    e = eps_V(phi_star, n)
    u_star = (phi_star / (1.0 + phi_star))**n
    return A_s * 24.0 * np.pi**2 * e / u_star


# ---------- Report ----------------------------------------------------------

def report(Ne_values, n_values):
    for n in n_values:
        phi_end = find_phi_end(n)
        print(f"\n{'='*78}")
        print(f"  n = {n}   |   V(phi) = V0 * [phi/(1+phi)]^{n}")
        print(f"  phi_end = {phi_end:.4f}   (eps_V = 1)")
        print(f"{'='*78}")
        print(f"{'N_e':>5}  {'phi_*':>8}  {'eps_V':>11}  {'eta_V':>11}  "
              f"{'n_s':>8}  {'r':>8}  {'V0 [M_P^4]':>12}")
        print("-" * 78)
        for Ne in Ne_values:
            phi_star = find_phi_star(Ne, phi_end, n)
            e, h, n_s, r = observables(phi_star, n)
            V0 = compute_V0(phi_star, n)
            print(f"{Ne:>5}  {phi_star:>8.4f}  {e:>11.4e}  {h:>+11.4e}  "
                  f"{n_s:>8.4f}  {r:>8.4f}  {V0:>12.4e}")

    print()
    print("Note: V/V' = phi(1+phi)/n  =>  N = [phi^2/2 + phi^3/3] / n")
    print("      phi_* grows with n at fixed N_e  =>  larger eps_V, larger r")


# ---------- Scan & plot -----------------------------------------------------

def scan(n_values):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        have_mpl = True
    except ImportError:
        have_mpl = False
        print("matplotlib not found; saving data only.")

    Ne_array = np.arange(40, 71, 1)
    colors = cm.viridis(np.linspace(0.15, 0.85, len(n_values)))

    all_data = []
    for n in n_values:
        phi_end = find_phi_end(n)
        ns_arr, r_arr = [], []
        for Ne in Ne_array:
            phi_star = find_phi_star(Ne, phi_end, n)
            _, _, n_s, r = observables(phi_star, n)
            ns_arr.append(n_s)
            r_arr.append(r)
        all_data.append((n, Ne_array, np.array(ns_arr), np.array(r_arr)))

    # Save combined data file
    with open('ns_r_scan_n.dat', 'w') as fout:
        fout.write(f"# n   N_e   n_s   r\n")
        for n, Ne_arr, ns_arr, r_arr in all_data:
            for Ne, ns, r in zip(Ne_arr, ns_arr, r_arr):
                fout.write(f"{n:4d}  {Ne:6.1f}  {ns:.6f}  {r:.6f}\n")
            fout.write("\n")
    print("Wrote ns_r_scan_n.dat")

    if not have_mpl:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_ne_ns, ax_ne_r, ax_ns_r = axes

    for (n, Ne_arr, ns_arr, r_arr), col in zip(all_data, colors):
        lbl = f"$n={n}$"
        ax_ne_ns.plot(Ne_arr, ns_arr, color=col, label=lbl)
        ax_ne_r.plot(Ne_arr, r_arr,  color=col, label=lbl)
        ax_ns_r.plot(ns_arr, r_arr,  color=col, label=lbl)
        # Mark Ne = 50, 55, 60
        for Ne_mark in [50, 55, 60]:
            idx = np.where(Ne_arr == Ne_mark)[0]
            if len(idx):
                ax_ns_r.scatter(ns_arr[idx], r_arr[idx], color=col, s=30, zorder=5)

    # ACT P-ACT-LB band
    ns_act, dns_act = 0.9743, 0.0034
    for ax in [ax_ne_ns]:
        ax.axhspan(ns_act - dns_act, ns_act + dns_act,
                   alpha=0.15, color='purple', label='P-ACT-LB 1σ')
    for ax in [ax_ns_r]:
        ax.axvspan(ns_act - dns_act, ns_act + dns_act,
                   alpha=0.15, color='purple', label='P-ACT-LB 1σ')

    # BK18 r < 0.036 (2σ)
    for ax in [ax_ne_r, ax_ns_r]:
        ax.axhline(0.036, ls='--', color='red', lw=1.2, label='BK18 $r<0.036$')

    ax_ne_ns.set_xlabel(r'$N_e$');  ax_ne_ns.set_ylabel(r'$n_s$')
    ax_ne_r.set_xlabel(r'$N_e$');   ax_ne_r.set_ylabel(r'$r$')
    ax_ns_r.set_xlabel(r'$n_s$');   ax_ns_r.set_ylabel(r'$r$')

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(r'$V(\phi) = V_0\,[\phi/(1+\phi)]^n$, slow-roll observables',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('ns_r_scan_n.png', dpi=150)
    print("Wrote ns_r_scan_n.png")


# ---------- CLI -------------------------------------------------------------

def main():
    n_values = [2, 3, 4, 5, 6, 7]

    parser = argparse.ArgumentParser()
    parser.add_argument('--Ne', type=float, default=None)
    parser.add_argument('--scan', action='store_true')
    args = parser.parse_args()

    if args.scan:
        scan(n_values)
    elif args.Ne is not None:
        report([args.Ne], n_values)
    else:
        report([50, 55, 60], n_values)


if __name__ == '__main__':
    main()
