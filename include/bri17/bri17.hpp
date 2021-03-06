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

#include <Eigen/Dense>

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
   * @param shape number of cells in each direction
   * @param L size of the grid in each direction (arbitrary units of length)
   */
  CartesianGrid(std::array<int, DIM> shape, std::array<T, DIM> L)
      : shape{shape},
        L{L},
        size{std::reduce(shape.cbegin(), shape.cend(), int{1},
                         std::multiplies())} {}

  /** Return a string representation of this object. */
  std::string repr() const {
    std::ostringstream stream;
    stream << "CartesianGrid<" << typeid(T).name() << "," << DIM << ">{shape={";
    for (auto n : shape) stream << n << ",";
    stream << "},L={";
    for (auto x : L) stream << x << ",";
    stream << "}}";
    return stream.str();
  }

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
   * @return array of node indices
   */
  std::array<int, num_nodes_per_cell> get_cell_nodes(int cell) const {
    std::array<int, num_nodes_per_cell> nodes;
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
    return nodes;
  }
};

/** Print the grid to the specified `ostream`. */
template <typename T, int DIM>
std::ostream &operator<<(std::ostream &os, const CartesianGrid<T, DIM> &grid) {
  return os << grid.repr();
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

  /** Return a string representation of this object. */
  std::string repr() const {
    std::ostringstream stream;
    stream << "Hooke<" << typeid(T).name() << "," << DIM << ">{mu=" << mu
           << ",nu=" << nu << ",grid=" << grid << std::endl;
    return stream.str();
  }

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
      c[i] = cos(alpha);
      s[i] = sin(alpha) * grid.shape[i] / grid.L[i];
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

  /**
   * Compute the strains induced by the specified eigenstresses.
   *
   * The eigenstresses `τ[n, i, j]` are constant in each cell n. They induce the
   * average strains `ε[n, i, j]`.
   *
   * This method computes the **opposite** of the induced strain!
   *
   * @param k multi-index of the Fourier component
   * @param tau the `k`-th Fourier component of the eigenstress `τ`,
   *            `τ^[k, :, :]`
   * @param eta the `k`-th Fourier component of `-ε`, `-ε^[k, :, :]`
   *            (output parameter).
   */
  void modal_eigenstress_to_opposite_strain(int const *k,
                                            std::complex<T> const *tau,
                                            std::complex<T> *eta) const {
    using Vector = Eigen::Matrix<std::complex<T>, DIM, 1>;
    using Matrix = Eigen::Matrix<std::complex<T>, DIM, DIM>;
    constexpr T const sqrt2 = std::numbers::sqrt2_v<T>;
    constexpr std::complex<T> zero{};
    constexpr int const sym = DIM == 2 ? 3 : 6;
    Vector B{};
    modal_strain_displacement(k, B.data());
    Matrix K{};
    modal_stiffness(k, K.data());
    Matrix tau_mat;
    bool null_frequency = false;
    if constexpr (DIM == 2) {
      // clang-format off
      tau_mat <<         tau[0], tau[2] / sqrt2,
                 tau[2] / sqrt2,         tau[1];
      // clang-format on
      null_frequency = (k[0] == 0) && (k[1] == 0);
    } else if constexpr (DIM == 3) {
      // clang-format off
      tau_mat <<         tau[0], tau[5] / sqrt2, tau[4] / sqrt2,
                 tau[5] / sqrt2,         tau[1], tau[3] / sqrt2,
                 tau[4] / sqrt2, tau[3] / sqrt2,         tau[2];
      // clang-format on
      null_frequency = (k[0] == 0) && (k[1] == 0) && (k[2] == 0);
    }
    if (null_frequency) {
      for (int i = 0; i < sym; i++) eta[i] = zero;
      return;
    }
    Vector rhs = tau_mat * B.conjugate();
    Vector u = K.llt().solve(rhs);
    Matrix eta_mat = 0.5 * (B * u.transpose() + u * B.transpose());
    if constexpr (DIM == 2) {
      eta[0] = eta_mat(0, 0);
      eta[1] = eta_mat(1, 1);
      eta[2] = sqrt2 * eta_mat(0, 1);
    } else if constexpr (DIM == 3) {
      eta[0] = eta_mat(0, 0);
      eta[1] = eta_mat(1, 1);
      eta[2] = eta_mat(2, 2);
      eta[3] = sqrt2 * eta_mat(1, 2);
      eta[4] = sqrt2 * eta_mat(2, 0);
      eta[5] = sqrt2 * eta_mat(0, 1);
    }
  }
};

/** Print the grid to the specified `ostream`. */
template <typename T, int DIM>
std::ostream &operator<<(std::ostream &os, const Hooke<T, DIM> &hooke) {
  return os << hooke.repr();
}

}  // namespace bri17
