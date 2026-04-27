# PBHInf

A lightweight C++17 numerical pipeline for computing the primordial scalar
power spectrum in single-field inflation, with a focus on **ultra-slow-roll
(USR) phases** that produce primordial black holes (PBHs) and stochastic
gravitational wave backgrounds.

## What it does

Given an inflaton potential $V(\phi)$, PBHInf solves:

1. The **background** equations of motion in e-folds $N$ (Klein-Gordon +
   Friedmann), tracking the field $\phi$, its cosmic-time derivative
   $\dot\phi$, the Hubble rate $H$, the slow-roll parameters $\epsilon$
   and $\eta_H$, and the conformal time $\eta$.
2. The **Mukhanov-Sasaki equation** for the curvature perturbation
   $\mathcal{R}_k$ directly (avoiding the $z''/z$ form for numerical
   stability across USR transitions).
3. Outputs the **scalar power spectrum** $\mathcal{P}_\zeta(k)$ across many
   decades in $k$, suitable for downstream PBH abundance and SIGW
   calculations.

The pipeline targets single-field models with a bump or inflection point
that amplifies $\mathcal{P}_\zeta$ from $\mathcal{O}(10^{-9})$ at CMB
scales to $\mathcal{O}(10^{-2})$ at sub-horizon scales relevant for PBH
formation.


## Repository structure

```
PBHInf/
├── src/
│   ├── model.hpp         # potential V(phi), V'(phi), parameters
│   ├── background.cpp    # background ODE integrator (RK4 in N)
│   ├── perturbation.cpp  # Mukhanov-Sasaki solver
│   ├── rescale.cpp       # converts code-unit log_k to physical Mpc^-1
│   ├── fpbh.py           # PBH abundance estimator (Press-Schechter)
│   └── Makefile
├── analytic/
│   ├── analytic.py         # closed-form slow-roll computations
│   ├── analytic_general.py # generic potential slow-roll machinery
│   ├── ns.py               # spectral index n_s analytical formula
│   └── r.py                # tensor-to-scalar ratio r analytical formula
├── .gitignore
└── README.md
```

## Requirements

- A C++17 compiler (GCC ≥ 8 or Clang ≥ 10)
- GNU Make
- Python 3.10+ with NumPy and Matplotlib (for the analytic scripts and
  post-processing)

## Build and run

### Quickstart

```bash
cd src
make
./background           # produces background.dat
./perturbation         # produces power_spectrum.dat
./rescale              # rescales log_k to physical Mpc^-1, writes power_spectrum_phys.dat
```

### Detailed steps

#### 1. Edit the model

Open `src/model.hpp`. The potential is defined as:

```cpp
inline constexpr double V0    = 0.79e-10;
inline constexpr double A     = 1.873137e-3;
inline constexpr double phi0  = 5.0;
inline constexpr double sigma = 1.59e-2;

inline double V(double phi) {
    const double f = phi * phi / ((1.0 + phi) * (1.0 + phi));
    const double dphi = phi - phi0;
    const double u = 0.5 * dphi * dphi / (sigma * sigma);
    const double g = 1.0 + A * std::exp(-u);
    return V0 * f * g;
}
```

This is the polynomial $\alpha$-attractor potential

$$V(\phi) = V_0 \, \frac{\phi^2}{(1+\phi)^2} \left[1 + A \exp\left(-\frac{(\phi-\phi_0)^2}{2\sigma^2}\right)\right]$$

with a Gaussian bump that triggers USR. Modify `V0`, `A`, `phi0`, `sigma`
or replace `V(phi)` and `dV(phi)` entirely for other models.

#### 2. Compile

```bash
cd src
make
```

This produces three executables: `background`, `perturbation`, `rescale`.

To rebuild from scratch:

```bash
make clean && make
```

#### 3. Run the background solver

```bash
./background
```

Output (terminal):

```
Inflation ended: N_tot = 89.6751  phi = 0.490769  dphi = -2.92684e-06  epsilon = 1.00036
```

Output file: `background.dat` with columns

```
N  phi  dphi  epsilon  logH  etaH  eta
```

where `dphi` is $\dot\phi$ (cosmic-time derivative), `etaH` is the
slow-roll parameter $\eta_H = -\ddot\phi / (H \dot\phi)$, and `eta` is
the conformal time.

#### 4. Run the perturbation solver

```bash
./perturbation
```

Output (terminal):

```
H_star    = 4.44216e-06
k_star    = 3.43026e+07  (log_k_star = 17.3507)
log_k range: [-6.39315, 67.679]
```

Output file: `power_spectrum.dat` with columns

```
k_code  P_zeta(k)
```

where `k_code` is in code units (proportional to physical $k$ but with
arbitrary normalization; see step 5).

#### 5. Rescale to physical units

```bash
./rescale
```

This identifies the pivot scale $k_\star = 0.05~\mathrm{Mpc}^{-1}$ in the
output and rescales the entire $k$ axis. Writes `power_spectrum_Mpc.dat`
with columns

```
k_Mpc^-1  P_zeta(k)
```

ready for plotting and PBH/SIGW post-processing.

#### 6. Plot

```bash
python3 -c "
import numpy as np, matplotlib.pyplot as plt
d = np.loadtxt('power_spectrum_phys.dat')
plt.loglog(d[:,0], d[:,1])
plt.xlabel(r'\$k\$ [Mpc\$^{-1}\$]')
plt.ylabel(r'\$\\mathcal{P}_\\zeta(k)\$')
plt.axhline(2.1e-9, ls='--', color='gray', label='CMB plateau')
plt.legend()
plt.tight_layout()
plt.savefig('power_spectrum.png', dpi=200)
"
```

#### 7. Estimate PBH abundance

```bash
python3 fpbh.py
```

This reads `power_spectrum_phys.dat`, applies the Press-Schechter formalism
with collapse threshold $\delta_c = 0.45$, and outputs $f_{\rm PBH}$ versus
PBH mass.

## Validating against analytical slow-roll

The `analytic/` directory contains Python scripts that compute $n_s$ and
$r$ from the analytical slow-roll expressions for the polynomial
$\alpha$-attractor potential.

```bash
cd analytic
python3 analytic.py        # full table of (N_e, phi_*, n_s, r)
python3 ns.py --Ne 55      # n_s at N_e = 55
python3 r.py  --Ne 55      # r at N_e = 55
```

Expected values for the default parameters at $N_e = 55$:

| Quantity | Analytical | Numerical |
| -------- | ---------- | --------- |
| $n_s$    | 0.9742     | 0.9760    |
| $r$      | 0.0138     | 0.0140    |

Agreement to $\sim 0.2\%$ in $n_s$ and $\sim 1\%$ in $r$ is expected; the
small discrepancy comes from sub-leading slow-roll corrections.

## Default model parameters

These produce a $P_\zeta$ peak at $k \sim 10^{12}~\mathrm{Mpc}^{-1}$
corresponding to PBHs of mass $M \sim 10^{-13}~M_\odot$ (asteroid-mass
range, viable as $f_{\rm PBH} \to 1$ dark matter).

| Parameter | Value          | Role                              |
| --------- | -------------- | --------------------------------- |
| $V_0$     | $0.79 \times 10^{-10}$ | overall scale (sets $A_s$) |
| $A$       | $0.001873$     | bump amplitude (controls peak)    |
| $\phi_0$  | $5.0$          | bump location                     |
| $\sigma$  | $1.59\times 10^{-2}$  | bump width                  |

The bump amplitude $A$ has an extreme cliff: $A < 0.001873$ gives
$\mathcal{P}_\zeta^{\rm peak} \ll 10^{-3}$ (insufficient for PBHs);
$A > 0.0018735$ traps the field on the bump and inflation never ends.
The window $[0.001873, 0.0018735]$ is where $f_{\rm PBH}$ ranges from
$10^{-8}$ to $\sim 1$.

## Numerical method summary

**Background**: RK4 in e-folds $N$ on the system

$$\frac{d\phi}{dN} = \frac{\dot\phi}{H}, \qquad \frac{d\dot\phi}{dN} = -3\dot\phi - \frac{V_\phi}{H}$$

with $H^2 = (\dot\phi^2/2 + V)/(3 M_{\rm pl}^2)$. Step size $dN = 10^{-4}$,
inflation termination at $\epsilon = 1$.

**Perturbations**: RK4 in $N$ on the curvature perturbation equation

$$\mathcal{R}_k'' + (3 - \epsilon + \eta_H)\mathcal{R}_k' + \frac{k^2}{a^2 H^2}\mathcal{R}_k = 0$$

(primes denote $d/dN$). Bunch-Davies initial conditions imposed at $k/aH = 100$
(deep sub-horizon). Power spectrum extracted as

$$\mathcal{P}_\zeta(k) = \frac{k^3}{2\pi^2}|\mathcal{R}_k|^2$$

evaluated at the end of inflation.


## License

GPL-3.0-or-later. See `LICENSE` for details.

## Author

**Aayush Randeep**
BS-MS Physics, IISER Bhopal
[github.com/aayush21-work](https://aayush21-work.github.io/)

**Prof. Rajib Saha** — supervision, theoretical framework, model development
Department of Physics, IISER Bhopal
[Faculty page](https://home.iiserb.ac.in/~rajib/)


