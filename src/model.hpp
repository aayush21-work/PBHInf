#pragma once
#include <cmath>

// Kallosh-Linde-Roest (2025), arXiv:2503.21030
// Einstein-frame potential from Jordan-frame chaotic inflation with (1+phi)R
// non-minimal coupling, plus a Gaussian bump for PBH formation:
//
//   V(phi) = V0 * phi^2 / (1 + phi)^2 * [1 + A * exp(-(phi-phi0)^2 / (2 sigma^2))]
//
// Predictions (no bump, A=0) at N* = 60:  n_s ~ 0.9735, r ~ 0.009  (fits P-ACT-LB)
//
// Note: The `m` parameter from the KKLT model is no longer present in the base
// potential — the scale is set entirely by V0. It has been removed to avoid a
// silently-unused variable.

inline constexpr double V0    = 5.6e-10;     // rescale to 5.6e-10 for A_s = 2.1e-9
inline constexpr double A     = 1.873137e-3;     // right on the cliff
inline constexpr double phi0  = 5.0;          // gives bump at N ~ 55
inline constexpr double sigma = 1.59e-2;      

inline double V(double phi) {
  const double onep = 1.0 + phi;
  const double f = (phi * phi) / (onep * onep);

  const double dphi = phi - phi0;
  const double u = 0.5 * dphi * dphi / (sigma * sigma);
  const double g = 1.0 + A * std::exp(-u);

  return V0 * f * g;
}

inline double dV(double phi) {
  const double onep = 1.0 + phi;
  const double onep2 = onep * onep;
  const double onep3 = onep2 * onep;

  const double f  = (phi * phi) / onep2;
  const double df = 2.0 * phi / onep3;        // d/dphi [ phi^2 / (1+phi)^2 ]

  const double s2 = sigma * sigma;
  const double dphi = phi - phi0;
  const double u = 0.5 * dphi * dphi / s2;
  const double up = dphi / s2;
  const double e = std::exp(-u);

  const double g = 1.0 + A * e;
  const double dg = -A * e * up;

  return V0 * (df * g + f * dg);
}

inline double d2V(double phi) {
  const double onep = 1.0 + phi;
  const double onep2 = onep * onep;
  const double onep3 = onep2 * onep;
  const double onep4 = onep2 * onep2;

  const double f   = (phi * phi) / onep2;
  const double df  = 2.0 * phi / onep3;
  const double d2f = 2.0 * (1.0 - 2.0 * phi) / onep4;   // d^2/dphi^2 [ phi^2/(1+phi)^2 ]

  const double s2 = sigma * sigma;
  const double dphi = phi - phi0;
  const double u = 0.5 * dphi * dphi / s2;
  const double up = dphi / s2;
  const double e = std::exp(-u);

  const double g = 1.0 + A * e;
  const double dg = -A * e * up;
  const double d2g = A * e * (up * up - 1.0 / s2);

  return V0 * (d2f * g + 2.0 * df * dg + f * d2g);
}
