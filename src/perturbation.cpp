// Mukhanov-Sasaki perturbation solver, evolved in e-folds N.
//
// Background rows: N, phi, dphi (cosmic-time), epsilon, logH.
// Comparisons with k are done in log-space: we track log_k and compare to
// N + logH rather than computing aH directly.
//
// MS equation in N-variable (derived from the conformal-time form
// v'' + (k^2 - z_tt/z) v = 0 by substituting dtau = dt/a and d/dt = H d/dN):
//
//   d^2 v / dN^2 + (1 - eps) dv/dN + [ k^2/(aH)^2 - (z_tt/z)/(aH)^2 ] v = 0
//
// where z_tt/z (z double-prime w.r.t. conformal time, divided by z) is
//   (z_tt/z)/a^2 = 2 H^2 + (5/2) dphi^2 + 2 dphi ddphi / H
//                  + (1/2) dphi^4 / H^2 - V''(phi)
// and ddphi is obtained from the KG equation ddphi = -3 H dphi - V'(phi).
//
// Bunch-Davies IC imposed deep inside the horizon at k = 50 aH:
//   v = 1/sqrt(2k),  dv/dt = -i k / a * 1/sqrt(2k)
// Converting dv/dt -> dv/dN = (1/H) dv/dt:
//   dvk_re / dN = 0
//   dvk_im / dN = - (k / (a H)) / sqrt(2k)
// And k/(aH) = exp(log_k - N - logH).

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include "model.hpp"

struct BgRow {
    double N;
    double phi;
    double dphi;     // d phi / d t (cosmic time)
    double eps;
    double logH;
};

static std::vector<BgRow> bg;

static void load_background(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        BgRow r;
        if (iss >> r.N >> r.phi >> r.dphi >> r.eps >> r.logH) bg.push_back(r);
    }
    std::cout << "Loaded " << bg.size() << " background rows, N_tot = "
              << bg.back().N << "\n";
}

// Linear interpolation of background at arbitrary N.
// Uses bisection since bg is sorted in N.
static BgRow interp_bg(double N) {
    if (N <= bg.front().N) return bg.front();
    if (N >= bg.back().N)  return bg.back();
    // Binary search
    std::size_t lo = 0, hi = bg.size() - 1;
    while (hi - lo > 1) {
        std::size_t mid = (lo + hi) / 2;
        if (bg[mid].N <= N) lo = mid; else hi = mid;
    }
    double t = (N - bg[lo].N) / (bg[hi].N - bg[lo].N);
    BgRow r;
    r.N    = N;
    r.phi  = bg[lo].phi  + t * (bg[hi].phi  - bg[lo].phi);
    r.dphi = bg[lo].dphi + t * (bg[hi].dphi - bg[lo].dphi);
    r.eps  = bg[lo].eps  + t * (bg[hi].eps  - bg[lo].eps);
    r.logH = bg[lo].logH + t * (bg[hi].logH - bg[lo].logH);
    return r;
}

// Find N such that k = fac * aH, i.e. log_k = N + logH + log(fac).
// Returns false if no crossing exists in the stored range.
static bool find_N_for_aH(double log_k, double fac, double& N_out) {
    double target = log_k - std::log(fac);   // we want N + logH == target
    // f(N) = N + logH(N) - target; monotonically increasing (logH drifts but
    // dominated by N), so bisect.
    auto f = [&](double N) {
        BgRow r = interp_bg(N);
        return r.N + r.logH - target;
    };
    double f_lo = f(bg.front().N);
    double f_hi = f(bg.back().N);
    if (f_lo > 0.0 || f_hi < 0.0) return false;
    double lo = bg.front().N, hi = bg.back().N;
    for (int i = 0; i < 80; ++i) {
        double mid = 0.5 * (lo + hi);
        double fm = f(mid);
        if (fm < 0.0) lo = mid; else hi = mid;
        if (hi - lo < 1e-10) break;
    }
    N_out = 0.5 * (lo + hi);
    return true;
}

// (z_tt/z)/a^2 evaluated from background quantities (cosmic-time form)
static inline double zpp_over_z_over_a2(double phi, double dphi, double H) {
    double ddphi = -3.0 * H * dphi - dV(phi);
    return 2.5 * dphi * dphi
         + 2.0 * dphi * ddphi / H
         + 2.0 * H * H
         + 0.5 * (dphi * dphi * dphi * dphi) / (H * H)
         - d2V(phi);
}

// RHS for the combined system (phi, dphi, vk_re, dvk_re, vk_im, dvk_im) in N.
// All derivatives denoted with suffix _N are d/dN.
struct State {
    double phi, dphi;
    double vre, dvre;
    double vim, dvim;
};

// Full RHS with current N available, so we can compute k^2/(aH)^2 safely.
static inline void rhs_N(double N, const State& s, double log_k,
                         State& ds) {
    double H    = std::sqrt((0.5 * s.dphi * s.dphi + V(s.phi)) / 3.0);
    double logH = std::log(H);
    double eps  = 0.5 * s.dphi * s.dphi / (H * H);

    // background: dphi/dN = dphi / H ; d(dphi)/dN = -3 dphi - V'/H
    ds.phi  = s.dphi / H;
    ds.dphi = -3.0 * s.dphi - dV(s.phi) / H;

    // k^2 / (aH)^2  and  (z_tt/z)/(aH)^2
    // log(k/(aH))^2 = 2 * (log_k - N - logH)
    double x         = log_k - N - logH;
    double k2_over_aH2 = std::exp(2.0 * x);

    double zpp_a2 = zpp_over_z_over_a2(s.phi, s.dphi, H);   // = (z_tt/z)/a^2
    double zpp_aH2 = zpp_a2 / (H * H);                      // divide by H^2

    // v'' = -(1 - eps) v' - [k^2/(aH)^2 - (z_tt/z)/(aH)^2] v
    double coeff = k2_over_aH2 - zpp_aH2;

    ds.vre  = s.dvre;
    ds.dvre = -(1.0 - eps) * s.dvre - coeff * s.vre;
    ds.vim  = s.dvim;
    ds.dvim = -(1.0 - eps) * s.dvim - coeff * s.vim;
}

// RK4 step in N
static inline void rk4_step(double& N, State& s, double dN, double log_k) {
    State k1, k2, k3, k4, tmp;
    rhs_N(N, s, log_k, k1);

    tmp.phi  = s.phi  + 0.5*dN*k1.phi;
    tmp.dphi = s.dphi + 0.5*dN*k1.dphi;
    tmp.vre  = s.vre  + 0.5*dN*k1.vre;
    tmp.dvre = s.dvre + 0.5*dN*k1.dvre;
    tmp.vim  = s.vim  + 0.5*dN*k1.vim;
    tmp.dvim = s.dvim + 0.5*dN*k1.dvim;
    rhs_N(N + 0.5*dN, tmp, log_k, k2);

    tmp.phi  = s.phi  + 0.5*dN*k2.phi;
    tmp.dphi = s.dphi + 0.5*dN*k2.dphi;
    tmp.vre  = s.vre  + 0.5*dN*k2.vre;
    tmp.dvre = s.dvre + 0.5*dN*k2.dvre;
    tmp.vim  = s.vim  + 0.5*dN*k2.vim;
    tmp.dvim = s.dvim + 0.5*dN*k2.dvim;
    rhs_N(N + 0.5*dN, tmp, log_k, k3);

    tmp.phi  = s.phi  + dN*k3.phi;
    tmp.dphi = s.dphi + dN*k3.dphi;
    tmp.vre  = s.vre  + dN*k3.vre;
    tmp.dvre = s.dvre + dN*k3.dvre;
    tmp.vim  = s.vim  + dN*k3.vim;
    tmp.dvim = s.dvim + dN*k3.dvim;
    rhs_N(N + dN, tmp, log_k, k4);

    s.phi  += (dN/6.0)*(k1.phi  + 2.0*k2.phi  + 2.0*k3.phi  + k4.phi);
    s.dphi += (dN/6.0)*(k1.dphi + 2.0*k2.dphi + 2.0*k3.dphi + k4.dphi);
    s.vre  += (dN/6.0)*(k1.vre  + 2.0*k2.vre  + 2.0*k3.vre  + k4.vre);
    s.dvre += (dN/6.0)*(k1.dvre + 2.0*k2.dvre + 2.0*k3.dvre + k4.dvre);
    s.vim  += (dN/6.0)*(k1.vim  + 2.0*k2.vim  + 2.0*k3.vim  + k4.vim);
    s.dvim += (dN/6.0)*(k1.dvim + 2.0*k2.dvim + 2.0*k3.dvim + k4.dvim);
    N += dN;
}

// Solve for a single k (log_k = log(k/aH_today-ish, same unit as aH stored)).
// Returns P_s. Writes NaN equivalent via returning negative on failure.
static double solve_for_logk(double log_k) {
    double N_start, N_end;
    // BD deep inside horizon (k = 50 aH); evaluate P_s far outside horizon
    // (k = aH/1000, ~7 e-folds after exit) to ensure USR modes have frozen.
    if (!find_N_for_aH(log_k, 50.0, N_start))  return -1.0;
    if (!find_N_for_aH(log_k, 1.0/1000.0, N_end)) return -1.0;
    if (N_end <= N_start) return -1.0;

    BgRow r0 = interp_bg(N_start);
    double k = std::exp(log_k);
    double aH_start = std::exp(N_start + r0.logH);

    State s;
    s.phi  = r0.phi;
    s.dphi = r0.dphi;
    // Bunch-Davies: v = 1/sqrt(2k),  dv/dt = -i k / a * 1/sqrt(2k)
    // dv/dN = dv/dt / H = -i (k/(aH)) / sqrt(2k)
    double inv_sqrt2k = 1.0 / std::sqrt(2.0 * k);
    s.vre  = inv_sqrt2k;
    s.dvre = 0.0;
    s.vim  = 0.0;
    s.dvim = -(k / aH_start) * inv_sqrt2k;   // k/(aH_start) = 50 by construction

    double N = N_start;
    const double dN = 1e-4;
    while (N < N_end) {
        double step = std::min(dN, N_end - N);
        rk4_step(N, s, step, log_k);
    }

    double H   = std::sqrt((0.5 * s.dphi * s.dphi + V(s.phi)) / 3.0);
    double eps = 0.5 * s.dphi * s.dphi / (H * H);
    double a   = std::exp(N);
    double vk2 = s.vre * s.vre + s.vim * s.vim;

    // P_s = (k^3 / (2 pi^2)) * |v_k|^2 / (2 a^2 epsilon)
    double Ps = (k * k * k) / (2.0 * M_PI * M_PI) * vk2 / (2.0 * a * a * eps);
    return Ps;
}

int main() {
    load_background("background.dat");

    // --- Compute CMB pivot scale k_star ---
    // k_star = (aH) at N = N_tot - 60, the epoch when the CMB pivot exits.
    // In our log-space storage:  log(k_star) = N_star + logH(N_star).
    double N_tot  = bg.back().N;
    double N_star = N_tot - 60.0;
    if (N_star < bg.front().N) {
        std::cerr << "ERROR: not enough inflation (N_tot = " << N_tot
                  << " < 60). Cannot define k_star.\n";
        return 1;
    }
    BgRow r_star      = interp_bg(N_star);
    double log_k_star = N_star + r_star.logH;
    double k_star     = std::exp(log_k_star);
    std::cout << "N_tot     = " << N_tot     << "\n"
              << "N_star    = " << N_star    << "\n"
              << "phi_star  = " << r_star.phi << "\n"
              << "H_star    = " << std::exp(r_star.logH) << "\n"
              << "k_star    = " << k_star    << "  (log_k_star = "
              << log_k_star << ")\n";

    // Pick log_k range from the available aH span.
    // BD is imposed at  N + logH = log_k - log(50)
    // End is imposed at N + logH = log_k + log(1000)
    // Both must lie comfortably inside the stored background range — use a
    // safety margin of 2 e-folds at each end to avoid BD/freeze-out artifacts.
    double aH_min_log = bg.front().N + bg.front().logH;
    double aH_max_log = bg.back().N  + bg.back().logH;
    const double safety = 2.0;  // e-folds of margin (need room for USR freeze-out)
    double log_k_min = aH_min_log + std::log(50.0) + safety;
    double log_k_max = aH_max_log - std::log(1000.0) - safety;
    std::cout << "log_k range: [" << log_k_min << ", " << log_k_max << "]\n";

    // V0 is now physical (0.79e-10 m_p^4 for Mishra-Sahni model), so no extra
    // normalization is applied here. P_s in output is the true curvature
    // power spectrum.  First column is k / k_star.

    std::ofstream out("power_spectrum.dat");
    out << std::setprecision(10);
    out << "# k/k_star P_s    (k_star = " << k_star << ")\n";

    for (double lk = log_k_min; lk <= log_k_max; lk += 0.1) {
        double Ps = solve_for_logk(lk);
        double k  = std::exp(lk);
        if (Ps < 0.0 || !std::isfinite(Ps)) {
            std::cerr << "Skipped log_k = " << lk << " (no crossing / NaN)\n";
            continue;
        }
        out << (k / k_star) << " " << Ps << "\n";
    }
    return 0;
}
