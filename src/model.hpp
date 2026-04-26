#pragma once
#include <cmath>

// Base: Kallosh-Linde-Roest (2025), arXiv:2503.21030
// Einstein-frame potential from Jordan-frame chaotic inflation with (1+phi)R
// non-minimal coupling, plus axion-monodromy-style sinusoidal modulation:
//
//   V(phi) = V0 * phi^2 / (1+phi)^2 * [1 + lambda * cos((phi - phi0) / f)]
//
// The modulation is active only in the window [phi0 - w, phi0 + w] via a
// smooth Hann envelope, so CMB scales (phi >> phi0) are untouched.
//
// Parameters:
//   lambda  -- modulation amplitude  (keep < 1 to avoid dV sign flip)
//   f       -- oscillation frequency scale (smaller f -> more oscillations -> more peaks)
//   phi0    -- centre of the modulation window
//   w       -- half-width of the Hann envelope (controls how many cycles are active)

inline constexpr double V0     = 5.6e-10;
inline constexpr double lambda = 1.0e-3;   // amplitude: small enough to preserve SR on CMB scales
inline constexpr double f_osc  = 0.20;  // period ~ 2*pi*f_osc in phi-space
inline constexpr double phi0   = 5.0;     // centre of modulation window (same as old Gaussian)
inline constexpr double w      = 0.5;     // Hann half-width: modulation lives in [phi0-w, phi0+w]

// Smooth Hann envelope: 1 inside window, tapers to 0 at edges
// This avoids a sharp kink in V and dV at the window boundary
inline double hann(double phi) {
    const double x = (phi - phi0) / w;
    if (std::abs(x) >= 1.0) return 0.0;
    const double c = std::cos(0.5 * M_PI * x);
    return c * c;   // Hann window: cos^2(pi*x/2)
}

inline double dhann(double phi) {
    const double x = (phi - phi0) / w;
    if (std::abs(x) >= 1.0) return 0.0;
    // d/dphi [ cos^2(pi*(phi-phi0)/(2w)) ] = -pi/(2w) * sin(pi*x)
    return -(M_PI / (2.0 * w)) * std::sin(M_PI * x);
}

inline double d2hann(double phi) {
    const double x = (phi - phi0) / w;
    if (std::abs(x) >= 1.0) return 0.0;
    return -(M_PI * M_PI / (2.0 * w * w)) * std::cos(M_PI * x);
}

// Modulation factor g(phi) = 1 + lambda * cos((phi-phi0)/f_osc) * hann(phi)
// The Hann envelope confines the oscillations to the window around phi0
inline double g(double phi) {
    return 1.0 + lambda * std::cos((phi - phi0) / f_osc) * hann(phi);
}

inline double dg(double phi) {
    const double dphi = phi - phi0;
    const double c  = std::cos(dphi / f_osc);
    const double s  = std::sin(dphi / f_osc);
    const double H  = hann(phi);
    const double dH = dhann(phi);
    // Product rule: d/dphi [ cos(...) * hann ] 
    return lambda * ((-s / f_osc) * H + c * dH);
}

inline double d2g(double phi) {
    const double dphi = phi - phi0;
    const double c   = std::cos(dphi / f_osc);
    const double s   = std::sin(dphi / f_osc);
    const double H   = hann(phi);
    const double dH  = dhann(phi);
    const double d2H = d2hann(phi);
    // d^2/dphi^2 [ cos(...) * H ]
    return lambda * (
        (-c / (f_osc * f_osc)) * H
        + 2.0 * (-s / f_osc) * dH
        + c * d2H
    );
}

// Base potential shape and its derivatives
inline double f_base(double phi) {
    const double onep = 1.0 + phi;
    return (phi * phi) / (onep * onep);
}

inline double df_base(double phi) {
    const double onep = 1.0 + phi;
    return 2.0 * phi / (onep * onep * onep);
}

inline double d2f_base(double phi) {
    const double onep = 1.0 + phi;
    return 2.0 * (1.0 - 2.0 * phi) / (onep * onep * onep * onep);
}

inline double V(double phi) {
    return V0 * f_base(phi) * g(phi);
}

inline double dV(double phi) {
    return V0 * (df_base(phi) * g(phi) + f_base(phi) * dg(phi));
}

inline double d2V(double phi) {
    return V0 * (d2f_base(phi) * g(phi)
               + 2.0 * df_base(phi) * dg(phi)
               + f_base(phi) * d2g(phi));
}
