#ifndef __BRI17_H_20200315__
#define __BRI17_H_20200315__

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace bri17 {

template <typename T>
T product(size_t n, T a[]) {
  T out{1};
  for (size_t i = 0; i < n; i++) {
    out *= a[i];
  }
  return out;
}

template <size_t DIM>
class CartesianGrid {
 public:
  const size_t num_nodes_per_cell;
  const size_t num_cells;
  // TODO How to make these arrays const?
  double L[DIM];
  size_t N[DIM];

  // Default value for L may have inconsistent dimension, but only the
  // DIM first elements will be considered.
  CartesianGrid(size_t N[], double L[] = {1., 1., 1.})
      : num_nodes_per_cell{1 << DIM}, num_cells{product(DIM, N)} {
    static_assert((DIM == 2) || (DIM == 3));
    for (size_t i = 0; i < DIM; i++) {
      this->L[i] = L[i];
      this->N[i] = N[i];
    }
  }

  size_t get_node_at(size_t i, size_t j) const {
    static_assert(DIM == 2, "this method expects a 2D grid");
    return i * N[1] + j;
  }

  size_t get_node_at(size_t i, size_t j, size_t k) const {
    static_assert(DIM == 3, "this method expects a 3D grid");
    return (i * N[1] + j) * N[2] + k;
  }

  void get_cell_nodes(const size_t cell, size_t nodes[]) const {
    static_assert(DIM != 3, "not implemented");
    if constexpr (DIM == 2) {
      const size_t i1 = cell / N[1];
      const size_t j1 = cell % N[1];
      const size_t i2 = i1 == N[0] - 1 ? 0 : i1 + 1;
      const size_t j2 = j1 == N[1] - 1 ? 0 : j1 + 1;
      nodes[0] = get_node_at(i1, j1);
      nodes[1] = get_node_at(i1, j2);
      nodes[2] = get_node_at(i2, j1);
      nodes[3] = get_node_at(i2, j2);
    } else if constexpr (DIM == 3) {
      // TODO Implement this case
    } else {
      throw std::logic_error("This should never occur");
    }
  }
};

template <size_t DIM>
std::ostream &operator<<(std::ostream &os, const CartesianGrid<DIM> &grid) {
  os << "CartesianGrid<" << DIM << ">={L=[";
  for (auto L_i : grid.L) {
    os << L_i << ",";
  }
  os << "],N=[";
  for (auto N_i : grid.N) {
    os << N_i << ",";
  }
  return os << "]}";
}

template <size_t DIM>
class Hooke {
 public:
  const double mu;
  const double nu;
  const CartesianGrid<DIM> &grid;

  Hooke(double mu, double nu, CartesianGrid<DIM> &grid)
      : mu{mu}, nu{nu}, grid{grid} {};

  void modal_strain_displacement(
      size_t const *k, Eigen::Matrix<std::complex<double>, DIM, 1> &B) const {
    double h_inv[DIM];
    double c[DIM];
    double s[DIM];
    double sum_alpha = 0.;

    for (size_t i = 0; i < DIM; i++) {
      h_inv[i] = grid.N[i] / grid.L[i];
      const double alpha = M_PI * k[i] / grid.N[i];
      c[i] = cos(alpha);
      s[i] = sin(alpha);
    }

    std::complex prefactor{-2 * sin(sum_alpha), 2 * cos(sum_alpha)};

    if (DIM == 2) {
      B(0) << prefactor * h_inv[0] * s[0] * c[1];
      B(1) = prefactor * h_inv[1] * c[0] * s[1];
    } else if (DIM == 3) {
      B(0) = prefactor * h_inv[0] * s[0] * c[1] * c[2];
      B(1) = prefactor * h_inv[1] * c[0] * s[1] * c[2];
      B(2) = prefactor * h_inv[2] * c[0] * c[1] * s[2];
    } else {
      throw std::logic_error("this should never occur");
    }
  }

  void modal_stiffness(size_t const *k,
                       Eigen::Matrix<std::complex<double>, DIM, DIM> &K) const {
    // {phi, chi, psi}[i] = {?, ?, psi}(z_i) in the notation of [Bri17]
    double h_inv[DIM];
    double phi[DIM];
    double psi[DIM];
    double chi[DIM];
    for (size_t i = 0; i < DIM; i++) {
      h_inv[i] = grid.N[i] / grid.L[i];
      double beta = 2 * M_PI * k[i] / grid.N[i];
      phi[i] = 2 * (1 - cos(beta));
      chi[i] = (2 + cos(beta)) / 3;
      psi[i] = sin(beta);
    }

    const double scaling = mu / (1. - 2. * nu);
    if (DIM == 2) {
      const double H_00 = h_inv[0] * h_inv[0] * phi[0] * chi[1];
      K(0, 0) = scaling * H_00;
      K(0, 1) = scaling * h_inv[0] * h_inv[1] * psi[0] * psi[1];
      const double H_11 = h_inv[1] * h_inv[1] * chi[0] * phi[1];
      K(1, 1) = scaling * H_11;

      // Symmetrization
      K(1, 0) = K(0, 1);

      const double K_diag = mu * (H_00 + H_11);
      K(0, 0) += K_diag;
      K(1, 1) += K_diag;
    } else if (DIM == 3) {
      const double H_00 = h_inv[0] * h_inv[0] * phi[0] * chi[1] * chi[2];
      K(0, 0) = scaling * H_00;
      K(0, 1) = scaling * h_inv[0] * h_inv[1] * psi[0] * psi[1] * chi[2];
      K(0, 2) = scaling * h_inv[0] * h_inv[2] * psi[0] * chi[1] * psi[2];
      const double H_11 = h_inv[1] * h_inv[1] * chi[0] * phi[1] * chi[2];
      K(1, 1) = scaling * H_11;
      K(1, 2) = scaling * h_inv[1] * h_inv[2] * chi[0] * psi[1] * psi[2];
      const double H_22 = h_inv[2] * h_inv[2] * chi[0] * chi[1] * phi[2];
      K(2, 2) = scaling * H_22;

      K(1, 0) = K(0, 1);
      K(2, 0) = K(0, 2);
      K(2, 1) = K(1, 2);

      const double K_diag = mu * (H_00 + H_11 + H_22);
      K(0, 0) += K_diag;
      K(1, 1) += K_diag;
      K(2, 2) += K_diag;
    } else {
      throw std::logic_error("this should never occur");
    }
  }

  void modal_eigenstress_to_strain(
      size_t const *k, Eigen::Matrix<std::complex<double>, DIM, DIM> &tau,
      Eigen::Matrix<std::complex<double>, DIM, DIM> &eps) const {
    Eigen::Matrix<std::complex<double>, DIM, 1> B{};
    Eigen::Matrix<std::complex<double>, DIM, DIM> K{};
    modal_strain_displacement(k, B);
    modal_stiffness<DIM>(k, mu, nu, K);
    auto u = -K.fullPivLu().solve(tau * B);
    eps = 0.5 * (B * u.transpose() + u * B.transpose());
  }
};

}  // namespace bri17
#endif  // __BRI17_H_20200315__
