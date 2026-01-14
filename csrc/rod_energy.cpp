#include <cmath>
#include <cstddef>
#include <algorithm>

extern "C" {

// x is length 3N (xyzxyz...)
// grad_out is length 3N
// Energy terms use periodic (closed-loop) indexing.
void rod_energy_grad(
    int N,
    const double* x,
    double kb,
    double ks,
    double l0,
    double q,
    double kappa,
    double* energy_out,
    double* grad_out
) {
    const int M = 3*N;
    for (int i = 0; i < M; ++i) grad_out[i] = 0.0;
    double E = 0.0;

    auto idx = [N](int i) {
        // wrap point index into [0, N-1]
        int r = i % N;
        return (r < 0) ? (r + N) : r;
    };

    auto get = [&](int i, int d) -> double {
        return x[3*idx(i) + d];
    };
    auto addg = [&](int i, int d, double v) {
        grad_out[3*idx(i) + d] += v;
    };

    // 1) Bending: kb * ||x_{i+1} - 2 x_i + x_{i-1}||^2
    // Let b_i = x_{i+1} - 2 x_i + x_{i-1}. E += kb * b_i^2.
    // Gradient contributions (per coordinate):
    // dE/dx_{i-1} += 2 kb b_i
    // dE/dx_{i}   += 2 kb b_i * (-2)
    // dE/dx_{i+1} += 2 kb b_i
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double bim = get(i+1,d) - 2.0*get(i,d) + get(i-1,d);
            E += kb * bim * bim;
            const double c = 2.0 * kb * bim;
            addg(i-1, d, c);
            addg(i,   d, -2.0*c);
            addg(i+1, d, c);
        }
    }

    // 2) Stretching: ks * (||x_{i+1}-x_i|| - l0)^2 for each segment
    for (int i = 0; i < N; ++i) {
        double dx0 = get(i+1,0) - get(i,0);
        double dx1 = get(i+1,1) - get(i,1);
        double dx2 = get(i+1,2) - get(i,2);
        double r = std::sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2);
        r = std::max(r, 1e-12); // avoid division by zero
        double diff = r - l0;
        E += ks * diff * diff;

        // d/dr ks (r-l0)^2 = 2 ks (r-l0)
        // dr/dx_{i+1} = (x_{i+1}-x_i)/r, dr/dx_i = -(x_{i+1}-x_i)/r
        double coeff = 2.0 * ks * diff / r;
        addg(i+1,0,  coeff * dx0);
        addg(i+1,1,  coeff * dx1);
        addg(i+1,2,  coeff * dx2);
        addg(i,0,   -coeff * dx0);
        addg(i,1,   -coeff * dx1);
        addg(i,2,   -coeff * dx2);
    }

    // 3) Screened Coulomb between non-adjacent nodes:
    // U(r)= q^2 exp(-kappa r)/r
    // dU/dr = q^2 exp(-k r) * ( -1/r^2 - k/r )
    // grad_i = dU/dr * (x_i - x_j)/r
    // grad_j = -grad_i
    const double q2 = q*q;
    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
            // exclude neighbors along the chain (including the wrap neighbor)
            int dist = std::abs(j - i);
            if (dist == 1 || dist == N-1) continue;

            double rx0 = get(i,0) - get(j,0);
            double rx1 = get(i,1) - get(j,1);
            double rx2 = get(i,2) - get(j,2);
            double r = std::sqrt(rx0*rx0 + rx1*rx1 + rx2*rx2);
            r = std::max(r, 1e-12);

            double expkr = std::exp(-kappa * r);
            double U = q2 * expkr / r;
            E += U;

            double dUdr = q2 * expkr * ( -1.0/(r*r) - kappa/r );
            double coeff = dUdr / r; // multiply by (x_i-x_j)
            double gx0 = coeff * rx0;
            double gx1 = coeff * rx1;
            double gx2 = coeff * rx2;

            addg(i,0, gx0); addg(i,1, gx1); addg(i,2, gx2);
            addg(j,0,-gx0); addg(j,1,-gx1); addg(j,2,-gx2);
        }
    }

    *energy_out = E;
}

} // extern "C"
