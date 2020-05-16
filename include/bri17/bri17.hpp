/**
 * Matrix-free finite element method introduced in doi:10.1002/nme.5263
 *
 * This library provides classes to compute the modal stiffness
 * matrices and strain-displacement vectors in Fourier space for a
 * homogeneous, periodic unit-cell. Combined with a FFT library, these
 * can be used to compute the solution to any problem of homogeneous,
 * periodic linear elasticity.
 */

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

/** Return the product `a[0] * ... * a[n-1]`. */
template <typename T>
T product(size_t n, T a[]) {
  T out{1};
  for (size_t i = 0; i < n; i++) {
    out *= a[i];
  }
  return out;
}

/**
 * A rectangular grid with fixed spacing in each direction.
 *
 * @tparam DIM the number of spatial dimensions (must be 2 or 3)
 */
template <size_t DIM>
class CartesianGrid {
 public:
  /** Number of nodes per cell: `2 ** DIM`. */
  const size_t num_nodes_per_cell;

  /** Total number of cells: `N[0] * N[1] * ... * N[DIM-1]`. */
  const size_t num_cells;

  /** Size of the grid in each direction (arbitrary units of length). */
  double L[DIM];

  /** Number of cells in each direction. */
  size_t N[DIM];

  /**
   * @param N number of cells in each direction
   * @param L size of the grid in each direction (arbitrary units of length)
   */
  CartesianGrid(size_t N[], double L[])
      : num_nodes_per_cell{1 << DIM}, num_cells{product(DIM, N)} {
    static_assert((DIM == 2) || (DIM == 3));
    for (size_t i = 0; i < DIM; i++) {
      this->L[i] = L[i];
      this->N[i] = N[i];
    }
  }

  /**
   * Return the index of the node located at <tt>[i, j]</tt>.
   *
   * This method cannot be called with a 3D grid (this condition is
   * checked at compile time). Nodes numbering follows the row-major
   * order convention.
   */
  size_t get_node_at(size_t i, size_t j) const {
    static_assert(DIM == 2, "this method expects a 2D grid");
    return i * N[1] + j;
  }

  /**
   * Return the index of the node located at <tt>[i, j, k]</tt>.
   *
   * This method cannot be called with a 2D grid (this condition is
   * checked at compile time). Nodes numbering follows the row-major
   * order convention.
   */
  size_t get_node_at(size_t i, size_t j, size_t k) const {
    static_assert(DIM == 3, "this method expects a 3D grid");
    return (i * N[1] + j) * N[2] + k;
  }

  /**
   * Return the indices of the vertices of a specific cell.
   *
   * Numbering of vertices in 2D
   *
   * ```
   * 2────4
   * │    │
   * │    │
   * 1────3
   * ```
   *
   *
   * Numbering of vertices in 3D
   *
   * ```
   *      4────────8
   *     ╱│       ╱│
        ╱ │      ╱ │
   *   ╱  │     ╱  │
   *  ╱   │    ╱   │
   * 2────────6 ───7
   * │   ╱3   │   ╱
   * │  ╱     │  ╱
     │ ╱      │ ╱
   * │╱       │╱
   * 1────────5
   * ```
   *
   * @param cell index of the cell (row-major order)
   * @param nodes array of node indices (return parameter)
   *
   * @todo Use move semantics?
   */
  void get_cell_nodes(const size_t cell, size_t nodes[]) const {
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
      const size_t k1 = cell % N[2];
      const size_t ij1 = cell / N[2];
      const size_t j1 = ij1 % N[1];
      const size_t i1 = ij1 / N[1];
      const size_t i2 = i1 == N[0] - 1 ? 0 : i1 + 1;
      const size_t j2 = j1 == N[1] - 1 ? 0 : j1 + 1;
      const size_t k2 = k1 == N[2] - 1 ? 0 : k1 + 1;
      nodes[0] = get_node_at(i1, j1, k1);
      nodes[1] = get_node_at(i1, j1, k2);
      nodes[2] = get_node_at(i1, j2, k1);
      nodes[3] = get_node_at(i1, j2, k2);
      nodes[4] = get_node_at(i2, j1, k1);
      nodes[5] = get_node_at(i2, j1, k2);
      nodes[6] = get_node_at(i2, j2, k1);
      nodes[7] = get_node_at(i2, j2, k2);
    } else {
      throw std::logic_error("This should never occur");
    }
  }
};

/** Print the grid to the specified `ostream`. */
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

/**
 * Implementation of the results of [Bri17] per se.
 *
 * This class provides methods to compute the modal strain-displacement and
 * stiffness matrices.
 */
template <size_t DIM>
class Hooke {
 public:
  /** The shear modulus of the material. */
  const double mu;

  /** The Poisson ratio of the material. */
  const double nu;

  /** Geometric description of the underlying FE grid. */
  const CartesianGrid<DIM> &grid;

  /**
   * @param mu shear modulus
   * @param nu Poisson ratio
   * @param grid the FE grid
   */
  Hooke(double mu, double nu, CartesianGrid<DIM> &grid)
      : mu{mu}, nu{nu}, grid{grid} {};

  /**
   * Compute modal strain-displacement vector for specified spatial frequency.
   *
   * @param k the multi-index in the frequency domain
   * @param B the strain-displacement vector `B^[k, :]` (output parameter)
   */
  void modal_strain_displacement(
      size_t const *k, Eigen::Matrix<std::complex<double>, DIM, 1> &B) const {
    double c[DIM];
    double s[DIM];
    double sum_alpha = 0.;

    for (size_t i = 0; i < DIM; i++) {
      const double alpha = M_PI * k[i] / grid.N[i];
      sum_alpha += alpha;
      c[i] = cos(alpha) * grid.L[i] / grid.N[i];
      s[i] = sin(alpha);
    }

    std::complex prefactor{-2 * sin(sum_alpha), 2 * cos(sum_alpha)};

    if constexpr (DIM == 2) {
      B(0) = prefactor * s[0] * c[1];
      B(1) = prefactor * c[0] * s[1];
    } else if constexpr (DIM == 3) {
      B(0) = prefactor * s[0] * c[1] * c[2];
      B(1) = prefactor * c[0] * s[1] * c[2];
      B(2) = prefactor * c[0] * c[1] * s[2];
    } else {
      throw std::logic_error("this should never occur");
    }
  }

  /**
   * Compute modal stiffness matrix for specified spatial frequency.
   *
   * @param k the multi-index in the frequency domain
   * @param K the stiffness matrix `K^[k, :, :]` (output parameter)
   */
  void modal_stiffness(size_t const *k,
                       Eigen::Matrix<std::complex<double>, DIM, DIM> &K) const {
    // In the notation of [Bri17, see Eq. (B.17)]
    //
    // phi[i] = phi(z_i) / h_i
    // chi[i] = chi(z_i) * h_i
    // psi[i] = psi(z_i)
    //
    // Which simplifies the expression of H_k (there are no h_i's).
    // Note that H_k is multiplied by the cell volume, so that the
    // modal_stiffness is the true modal stiffness
    double phi[DIM];
    double psi[DIM];
    double chi[DIM];
    for (size_t i = 0; i < DIM; i++) {
      double h = grid.L[i] / grid.N[i];
      double beta = 2 * M_PI * k[i] / grid.N[i];
      phi[i] = 2 * (1 - cos(beta)) / h;
      chi[i] = (2 + cos(beta)) * h / 3;
      psi[i] = sin(beta);
    }

    const double scaling = mu / (1. - 2. * nu);
    if constexpr (DIM == 2) {
      const double H_00 = phi[0] * chi[1];
      K(0, 0) = scaling * H_00;
      K(0, 1) = scaling * psi[0] * psi[1];
      const double H_11 = chi[0] * phi[1];
      K(1, 1) = scaling * H_11;

      // Symmetrization
      K(1, 0) = K(0, 1);

      const double K_diag = mu * (H_00 + H_11);
      K(0, 0) += K_diag;
      K(1, 1) += K_diag;
    } else if constexpr (DIM == 3) {
      const double H_00 = phi[0] * chi[1] * chi[2];
      K(0, 0) = scaling * H_00;
      K(0, 1) = scaling * psi[0] * psi[1] * chi[2];
      K(0, 2) = scaling * psi[0] * chi[1] * psi[2];
      const double H_11 = chi[0] * phi[1] * chi[2];
      K(1, 1) = scaling * H_11;
      K(1, 2) = scaling * chi[0] * psi[1] * psi[2];
      const double H_22 = chi[0] * chi[1] * phi[2];
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

  /**
   * Compute the strains induced by the specified eigenstresses.
   *
   * The eigenstresses `τ[n, i, j]` are constant in each cell n. They induce the
   * strains `ε[n, i, j]`.
   *
   * **Warning: this method has not been tested.**
   *
   * @param k multi-index of the Fourier component
   * @param tau the `k`-th Fourier component of the eigenstress `τ`,
   * `τ^[k, :, :]`
   * @param eps the `k`-th Fourier component of `ε`, `ε^[k, :, :]` (output
   * parameter)
   */
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
