import numpy as np
from scipy.integrate import simpson
from scipy.special import erfc
import matplotlib.pyplot as plt
import sys

def run_pbh_analysis(filename):
    data = np.loadtxt(filename, comments='#')
    k_arr = data[:, 0]
    Ps_arr = data[:, 1]

    delta_c = 0.45
    gamma = 0.2
    g_star = 106.75

    M_pbh = 30.0 * (gamma / 0.2) * (g_star / 106.75) ** (-1.0 / 6.0) * (k_arr / 1e6) ** (-2.0)

    ln_k = np.log(k_arr)
    sigma2_arr = np.zeros_like(k_arr)

    for i, k_s in enumerate(k_arr):
        R = 1.0 / k_s
        kR = k_arr * R
        W = np.exp(-kR**2 / 2)
        integrand = (16.0 / 81.0) * Ps_arr * (kR**4) * (W**2)
        sigma2_arr[i] = simpson(integrand, x=ln_k)

    sigma2_arr = np.maximum(sigma2_arr, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        arg = np.where(sigma2_arr > 0, delta_c / np.sqrt(2.0 * sigma2_arr), np.inf)

    beta_arr = 0.5 * erfc(arg)

    f_pbh_arr = 2.4e8 * beta_arr * (g_star / 106.75) ** (-0.25) * M_pbh ** (-0.5)

    ps_peak_idx = np.argmax(Ps_arr)
    fpbh_peak_idx = np.argmax(f_pbh_arr)

    print("=" * 50)
    print(f"PBH RESULTS: {filename}")
    print("=" * 50)

    print("\n[Power spectrum peak]")
    print(f"k       = {k_arr[ps_peak_idx]:.4e}")
    print(f"P_R     = {Ps_arr[ps_peak_idx]:.4e}")
    print(f"M_PBH   = {M_pbh[ps_peak_idx]:.4e} M_sun")
    print(f"sigma^2 = {sigma2_arr[ps_peak_idx]:.4e}")
    print(f"beta    = {beta_arr[ps_peak_idx]:.4e}")
    print(f"f_PBH   = {f_pbh_arr[ps_peak_idx]:.4e}")

    print("\n[f_PBH peak]")
    print(f"k       = {k_arr[fpbh_peak_idx]:.4e}")
    print(f"P_R     = {Ps_arr[fpbh_peak_idx]:.4e}")
    print(f"M_PBH   = {M_pbh[fpbh_peak_idx]:.4e} M_sun")
    print(f"sigma^2 = {sigma2_arr[fpbh_peak_idx]:.4e}")
    print(f"beta    = {beta_arr[fpbh_peak_idx]:.4e}")
    print(f"f_PBH   = {f_pbh_arr[fpbh_peak_idx]:.4e}")
    print("=" * 50)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Primordial Black Hole Formation", fontsize=14)

    axes[0].loglog(k_arr, Ps_arr)
    axes[0].set_title("P_R(k)")
    axes[0].set_xlabel("k [Mpc^-1]")
    axes[0].set_ylabel("P_R")
    axes[0].grid(True, which='both')

    axes[1].loglog(M_pbh, beta_arr)
    axes[1].set_title("beta(M)")
    axes[1].set_xlabel("M [M_sun]")
    axes[1].set_ylabel("beta")
    axes[1].set_xlim([1e-11,1e-7])
    axes[1].grid(True, which='both')

    axes[2].loglog(M_pbh, f_pbh_arr)
    axes[2].axhline(1.0, linestyle='--')
    axes[2].set_title("f_PBH(M)")
    axes[2].set_xlabel("M [M_sun]")
    axes[2].set_ylabel("f_PBH")
    axes[2].set_xlim([1e-11,1e-7])
    axes[2].grid(True, which='both')

    plt.tight_layout()
    plt.savefig("pbh_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "power_spectrum_Mpc.dat"
    run_pbh_analysis(fname)
