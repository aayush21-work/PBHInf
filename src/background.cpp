// Background inflationary evolution in e-folds N.
//
// State: y = (phi, dphi)  where dphi = d phi / d t  (cosmic-time derivative)
// Evolution variable: N = ln(a/a0)
//
// Friedmann: H^2 = (1/3) [ (1/2) dphi^2 + V(phi) ]   (in units 8 pi G = 1,
// M_pl=1) d phi / dN  = dphi / H d dphi / dN = (-3 H dphi - V'(phi)) / H = -3
// dphi - V'(phi)/H
//
// We store N, phi, dphi, log_H directly (no raw 'a' to avoid overflow).
//
// Stop condition: epsilon_H = dphi^2 / (2 H^2) >= 1.

#include "model.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

static inline double H_of(double phi, double dphi) {
  return std::sqrt((0.5 * dphi * dphi + V(phi)) / 3.0);
}

static inline double epsilon_of(double phi, double dphi) {
  double h = H_of(phi, dphi);
  return 0.5 * dphi * dphi / (h * h);
}

// RHS of (dphi/dN, d(dphi)/dN)
static inline void rhs(double phi, double dphi, double &fp, double &fdp) {
  double h = H_of(phi, dphi);
  fp = dphi / h;
  fdp = -3.0 * dphi - dV(phi) / h;
}

static inline double etaH_of(double phi, double dphi) {
  double h = H_of(phi, dphi);
  return 3.0 + dV(phi) / (h * dphi);
}

int main() {

  double phi = 7.5;
  
  double dphi = -dV(phi) / (3.0 * H_of(phi, 0.0)); 
  // Iterate once for a tighter attractor value:
  dphi = -dV(phi) / (3.0 * H_of(phi, dphi));

  double N = 0.0;

  const double dN = 1e-4;
  const int write_every = 10;
  const double N_max = 1000; // hard safety cap

  std::ofstream out("background.dat");
  out << std::setprecision(15);

  // Write header row (commented) and initial row
  out << "# N phi dphi epsilon logH eta\n";
  out << N << " " << phi << " " << dphi << " " << epsilon_of(phi, dphi) << " "
      << std::log(H_of(phi, dphi)) << " " << etaH_of <<  "\n";

  int step = 0;
  while (epsilon_of(phi, dphi) < 1.0 && N < N_max) {
    // RK4 in N
    double k1p, k1d, k2p, k2d, k3p, k3d, k4p, k4d;

    rhs(phi, dphi, k1p, k1d);
    rhs(phi + 0.5 * dN * k1p, dphi + 0.5 * dN * k1d, k2p, k2d);
    rhs(phi + 0.5 * dN * k2p, dphi + 0.5 * dN * k2d, k3p, k3d);
    rhs(phi + dN * k3p, dphi + dN * k3d, k4p, k4d);

    phi += (dN / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
    dphi += (dN / 6.0) * (k1d + 2.0 * k2d + 2.0 * k3d + k4d);
    N += dN;
    ++step;

    if (step % write_every == 0) {
      out << N << " " << phi << " " << dphi << " " << epsilon_of(phi, dphi)
          << " " << std::log(H_of(phi, dphi)) << " " << etaH_of(phi,dphi) << "\n";
    }
  }
  // Ensure the final point is written
  out << N << " " << phi << " " << dphi << " " << epsilon_of(phi, dphi) << " "
      << std::log(H_of(phi, dphi)) << " " << etaH_of
      << "\n";

  std::cout << "Inflation ended: N_tot = " << N << "  phi = " << phi
            << "  dphi = " << dphi << "  epsilon = " << epsilon_of(phi, dphi)
            << "\n";
  return 0;
}
