/**
 * Matrix-free finite element method introduced in doi:10.1002/nme.5263
 *
 * This library provides classes to compute the modal stiffness
 * matrices and strain-displacement vectors in Fourier space for a
 * homogeneous, periodic unit-cell. Combined with a FFT library, these
 * can be used to compute the solution to any problem of homogeneous,
 * periodic linear elasticity.
 */

#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <numbers>
#include <numeric>

#include <complex>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace bri17 {
/**
 * A rectangular grid with fixed spacing in each direction.
 *
 * @tparam T the scalar type
 * @tparam DIM the number of spatial dimensions (must be 2 or 3)
 */
template <typename T, int DIM>
requires(std::floating_point<T> &&
         ((DIM == 2) || (DIM == 3))) class CartesianGrid {
 public:
  /** Number of nodes per cell: `2 ** DIM`. */
  static constexpr int num_nodes_per_cell = 1 << DIM;

  /** Number of cells in each direction. */
  std::array<int, DIM> const shape;

  /** Size of the grid in each direction (arbitrary units of length). */
  std::array<T, DIM> const L;

  /** Total number of cells: `shape[0] * shape[1] * ... * shape[DIM-1]`. */
  int const size;

  /**
   * @param N number of cells in each direction
   * @param L size of the grid in each direction (arbitrary units of length)
   */
  CartesianGrid(std::array<int, DIM> N, std::array<T, DIM> L)
      : shape{N},
        L{L},
        size{std::reduce(N.cbegin(), N.cend(), int{1}, std::multiplies())}{}

  /**
   * Return the index of the node located at <tt>[i, j]</tt>.
   *
   * This method cannot be called with a 3D grid (this condition is
   * checked at compile time). Nodes numbering follows the row-major
   * order convention.
   */
  int get_node_at(int i, int j) const {
    static_assert(DIM == 2, "this method expects a 2D grid");
    return i * shape[1] + j;
  }

  /**
   * Return the index of the node located at <tt>[i, j, k]</tt>.
   *
   * This method cannot be called with a 2D grid (this condition is
   * checked at compile time). Nodes numbering follows the row-major
   * order convention.
   */
  int get_node_at(int i, int j, int k) const {
    static_assert(DIM == 3, "this method expects a 3D grid");
    return (i * shape[1] + j) * shape[2] + k;
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
  void get_cell_nodes(const int cell, int nodes[]) const {
    if constexpr (DIM == 2) {
      const int i1 = cell / shape[1];
      const int j1 = cell % shape[1];
      const int i2 = i1 == shape[0] - 1 ? 0 : i1 + 1;
      const int j2 = j1 == shape[1] - 1 ? 0 : j1 + 1;
      nodes[0] = get_node_at(i1, j1);
      nodes[1] = get_node_at(i1, j2);
      nodes[2] = get_node_at(i2, j1);
      nodes[3] = get_node_at(i2, j2);
    } else if constexpr (DIM == 3) {
      const int k1 = cell % shape[2];
      const int ij1 = cell / shape[2];
      const int j1 = ij1 % shape[1];
      const int i1 = ij1 / shape[1];
      const int i2 = i1 == shape[0] - 1 ? 0 : i1 + 1;
      const int j2 = j1 == shape[1] - 1 ? 0 : j1 + 1;
      const int k2 = k1 == shape[2] - 1 ? 0 : k1 + 1;
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
template <typename T, int DIM>
std::ostream &operator<<(std::ostream &os, const CartesianGrid<T, DIM> &grid) {
  os << "CartesianGrid<" << DIM << ">={L=[";
  for (auto L_i : grid.L) {
    os << L_i << ",";
  }
  os << "],shape=[";
  for (auto N_i : grid.shape) {
    os << N_i << ",";
  }
  return os << "]}";
}

/**
 * Implementation of the results of [Bri17] per se.
 *
 * This class provides methods to compute the modal strain-displacement and
 * stiffness matrices.
 *
 * @tparam T the scalar type
 * @tparam DIM the number of spatial dimensions (must be 2 or 3)
 */
template <typename T, int DIM>
requires(std::floating_point<T> && ((DIM == 2) || (DIM == 3))) class Hooke {
 public:
  /** The shear modulus of the material. */
  T const mu;

  /** The Poisson ratio of the material. */
  T const nu;

  /** Geometric description of the underlying FE grid. */
  CartesianGrid<T, DIM> const grid;

  /**
   * @param mu shear modulus
   * @param nu Poisson ratio
   * @param grid the FE grid
   */
  Hooke(T mu, T nu, CartesianGrid<T, DIM> &grid)
      : mu{mu}, nu{nu}, grid{grid} {};

  /**
   * Compute modal strain-displacement vector for specified spatial frequency.
   *
   * The output parameter `B` must be a preallocated array of size `DIM`.
   *
   * @param k the multi-index in the frequency domain
   * @param B the strain-displacement vector `B^[k, :]` (output parameter)
   */
  void modal_strain_displacement(int const *k, std::complex<T> *B) const {
    T c[DIM];
    T s[DIM];
    T sum_alpha{};  // TODO Check that initializes to 0

    for (int i = 0; i < DIM; i++) {
      T alpha = std::numbers::pi_v<T> * k[i] / grid.shape[i];
      sum_alpha += alpha;
      c[i] = cos(alpha) * grid.L[i] / grid.shape[i];
      s[i] = sin(alpha);
    }

    std::complex<T> prefactor{-2 * sin(sum_alpha), 2 * cos(sum_alpha)};

    if constexpr (DIM == 2) {
      B[0] = prefactor * s[0] * c[1];
      B[1] = prefactor * c[0] * s[1];
    } else if constexpr (DIM == 3) {
      B[0] = prefactor * s[0] * c[1] * c[2];
      B[1] = prefactor * c[0] * s[1] * c[2];
      B[2] = prefactor * c[0] * c[1] * s[2];
    } else {
      throw std::logic_error("this should never occur");
    }
  }

  /**
   * Compute modal stiffness matrix for specified spatial frequency.
   *
   * The output parameter `K` must be a preallocated array of size
   * `DIM * DIM`.
   *
   * @param k the multi-index in the frequency domain
   * @param K the stiffness matrix `K^[k, :, :]` (output parameter)
   */
  void modal_stiffness(int const *k, std::complex<T> *K) const {
    // In the notation of [Bri17, see Eq. (B.17)]
    //
    // phi[i] = phi(z_i) / h_i
    // chi[i] = chi(z_i) * h_i
    // psi[i] = psi(z_i)
    //
    // Which simplifies the expression of H_k (there are no h_i's).
    // Note that H_k is multiplied by the cell volume, so that the
    // modal_stiffness is the true modal stiffness
    T phi[DIM];
    T psi[DIM];
    T chi[DIM];
    for (int i = 0; i < DIM; i++) {
      T h = grid.L[i] / grid.shape[i];
      T beta = 2 * std::numbers::pi_v<T> * k[i] / grid.shape[i];
      phi[i] = 2 * (1 - cos(beta)) / h / h;
      chi[i] = (2 + cos(beta)) / 3;
      psi[i] = sin(beta) / h;
    }

    const double scaling = mu / (1. - 2. * nu);
    if constexpr (DIM == 2) {
      auto H_00 = phi[0] * chi[1];
      auto H_11 = chi[0] * phi[1];
      auto K_diag = mu * (H_00 + H_11);
      K[0] = scaling * H_00 + K_diag;
      K[1] = scaling * psi[0] * psi[1];
      K[2] = K[1];
      K[3] = scaling * H_11 + K_diag;
    } else if constexpr (DIM == 3) {
      auto H_00 = phi[0] * chi[1] * chi[2];
      auto H_11 = chi[0] * phi[1] * chi[2];
      auto H_22 = chi[0] * chi[1] * phi[2];
      auto K_diag = mu * (H_00 + H_11 + H_22);
      K[0] = scaling * H_00 + K_diag;             // [0, 0]
      K[1] = scaling * psi[0] * psi[1] * chi[2];  // [0, 1]
      K[2] = scaling * psi[0] * chi[1] * psi[2];  // [0, 2]
      K[3] = K[1];                                // [1, 0]
      K[4] = scaling * H_11 + K_diag;             // [1, 1]
      K[5] = scaling * chi[0] * psi[1] * psi[2];  // [1, 2]
      K[6] = K[2];                                // [2, 0]
      K[7] = K[5];                                // [2, 1]
      K[8] = scaling * H_22 + K_diag;             // [2, 2]
    } else {
      throw std::logic_error("this should never occur");
    }
  }

  //  /**
  //   * Compute the strains induced by the specified eigenstresses.
  //   *
  //   * The eigenstresses `τ[n, i, j]` are constant in each cell n. They induce
  //   the
  //   * strains `ε[n, i, j]`.
  //   *
  //   * **Warning: this method has not been tested.**
  //   *
  //   * @param k multi-index of the Fourier component
  //   * @param tau the `k`-th Fourier component of the eigenstress `τ`,
  //   * `τ^[k, :, :]`
  //   * @param eps the `k`-th Fourier component of `ε`, `ε^[k, :, :]` (output
  //   * parameter)
  //   */
  //  void modal_eigenstress_to_strain(
  //      int const *k, Eigen::Matrix<std::complex<double>, DIM, DIM> &tau,
  //      Eigen::Matrix<std::complex<double>, DIM, DIM> &eps) const {
  //    Eigen::Matrix<std::complex<double>, DIM, 1> B{};
  //    Eigen::Matrix<std::complex<double>, DIM, DIM> K{};
  //    modal_strain_displacement(k, B);
  //    modal_stiffness<DIM>(k, mu, nu, K);
  //    auto u = -K.fullPivLu().solve(tau * B);
  //    eps = 0.5 * (B * u.transpose() + u * B.transpose());
  //  }
};

}  // namespace bri17
