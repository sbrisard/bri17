#include "bri17/bri17.h"
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fftw3.h>

const size_t MAX_DIM = 3;

template <size_t DIM>
class CartesianGrid {
 private:
  static size_t get_num_nodes_per_cell() {
    if (DIM == 2) {
      return 4;
    } else if (DIM == 3) {
      return 8;
    } else {
      throw std::logic_error("this should never occur");
    }
  }

  static size_t get_num_cells(size_t N[]) {
    if (DIM == 2) {
      return N[0] * N[1];
    } else if (DIM == 3) {
      return N[0] * N[1] * N[2];
    } else {
      throw std::logic_error("this should never occur");
    }
  }

 public:
  const size_t num_nodes_per_cell;
  const size_t num_cells;
  double mu;
  double nu;
  // TODO How to make these array const?
  double L[DIM];
  size_t N[DIM];

  CartesianGrid(double mu, double nu, double L[], size_t N[])
      : num_nodes_per_cell{get_num_nodes_per_cell()},
        num_cells{get_num_cells(N)} {
    this->mu = mu;
    this->nu = nu;
    static_assert((DIM == 2) || (DIM == 3));
    for (size_t i = 0; i < DIM; i++) {
      this->L[i] = L[i];
      this->N[i] = N[i];
    }
  }

  size_t get_node_at(size_t i, size_t j) {
    static_assert(DIM == 2, "this function should not be called with 3D grids");
    return i * N[1] + j;
  }

  size_t get_node_at(size_t i, size_t j, size_t k) {
    static_assert(DIM == 3, "this function should not be called with 2D grids");
    return (i * N[1] + j) * N[2] + k;
  }

  void get_cell_nodes(const size_t cell, size_t nodes[]) {
    static_assert(DIM != 3, "not implemented");
    if (DIM == 2) {
      const size_t i1 = cell / N[1];
      const size_t j1 = cell % N[1];
      const size_t i2 = i1 == N[0] - 1 ? 0 : i1 + 1;
      const size_t j2 = j1 == N[1] - 1 ? 0 : j1 + 1;
      nodes[0] = get_node_at(i1, j1);
      nodes[1] = get_node_at(i1, j2);
      nodes[2] = get_node_at(i2, j1);
      nodes[3] = get_node_at(i2, j2);
    } else if (DIM == 3) {
      // TODO Implement this case
    } else {
      throw std::logic_error("This should never occur");
    }
  }

  void modal_strain_displacement(
      size_t const *k, Eigen::Matrix<std::complex<double>, DIM, 1> &B) const;
  void modal_stiffness(size_t const *k,
                       Eigen::Matrix<std::complex<double>, DIM, DIM> &K) const;
  void modal_eigenstress_to_strain(
      size_t const *k, Eigen::Matrix<std::complex<double>, DIM, DIM> &tau,
      Eigen::Matrix<std::complex<double>, DIM, DIM> &eps) const;
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
void CartesianGrid<DIM>::modal_strain_displacement(
    size_t const *k, Eigen::Matrix<std::complex<double>, DIM, 1> &B) const {
  double h_inv[DIM];
  double c[DIM];
  double s[DIM];
  double sum_alpha = 0.;

  for (size_t i = 0; i < DIM; i++) {
    h_inv[i] = N[i] / L[i];
    const double alpha = M_PI * k[i] / N[i];
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

template <size_t DIM>
void CartesianGrid<DIM>::modal_stiffness(
    size_t const *k, Eigen::Matrix<std::complex<double>, DIM, DIM> &K) const {
  // {phi, chi, psi}[i] = {?, ?, psi}(z_i) in the notation of [Bri17]
  double h_inv[DIM];
  double phi[DIM];
  double psi[DIM];
  double chi[DIM];
  for (size_t i = 0; i < DIM; i++) {
    h_inv[i] = N[i] / L[i];
    double beta = 2 * M_PI * k[i] / N[i];
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

template <size_t DIM>
void CartesianGrid<DIM>::modal_eigenstress_to_strain(
    size_t const *k, Eigen::Matrix<std::complex<double>, DIM, DIM> &tau,
    Eigen::Matrix<std::complex<double>, DIM, DIM> &eps) const {
  Eigen::Matrix<std::complex<double>, DIM, 1> B{};
  Eigen::Matrix<std::complex<double>, DIM, DIM> K{};
  modal_strain_displacement(k, B);
  modal_stiffness(k, mu, nu, K);
  auto u = -K.fullPivLu().solve(tau * B);
  eps = 0.5 * (B * u.transpose() + u * B.transpose());
}

class FFTWComplexBuffer {
 public:
  fftw_complex *c_data;
  std::complex<double> *cpp_data;

  FFTWComplexBuffer(size_t n) {
    c_data = (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
    cpp_data = reinterpret_cast<std::complex<double> *>(c_data);
  }

  ~FFTWComplexBuffer() { fftw_free(c_data); }
};

template <size_t DIM>
class StiffnessMatrixFactory {
 public:
  const size_t ncells;
  const size_t ndofs;

 private:
  const CartesianGrid<DIM> grid;
  const FFTWComplexBuffer u;
  const FFTWComplexBuffer u_hat;
  const FFTWComplexBuffer Ku;
  const FFTWComplexBuffer Ku_hat;
  // TODO These members should be const
  fftw_plan dft_u[DIM];
  fftw_plan idft_Ku[DIM];

  static size_t num_cells(size_t N[]) {
    if (DIM == 2) {
      return N[0] * N[1];
    } else if (DIM == 3) {
      return N[0] * N[1] * N[2];
    } else {
      throw std::logic_error("this should never occur");
    }
  }

  void compute_Ku();

 public:
  StiffnessMatrixFactory(double mu, double nu, double L[], size_t N[])
      : ncells{num_cells(N)},
        ndofs{ncells * DIM},
        grid{mu, nu, L, N},
        u{ndofs},
        u_hat{ndofs},
        Ku{ndofs},
        Ku_hat{ndofs} {
    int N_[DIM];
    for (size_t i = 0; i < DIM; i++) N_[i] = N[i];
    for (size_t k = 0; k < DIM; k++) {
      size_t offset = k * ncells;
      dft_u[k] =
          fftw_plan_dft(DIM, N_, u.c_data + offset, u_hat.c_data + offset,
                        FFTW_FORWARD, FFTW_ESTIMATE);
      idft_Ku[k] =
          fftw_plan_dft(DIM, N_, Ku_hat.c_data + offset, Ku.c_data + offset,
                        FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }

  void run(Eigen::MatrixXcd &K);
};

template <size_t DIM>
void StiffnessMatrixFactory<DIM>::compute_Ku() {
  for (size_t i = 0; i < DIM; i++) fftw_execute(dft_u[i]);
  size_t k[DIM] = {0};
  Eigen::Matrix<std::complex<double>, DIM, DIM> K_k;
  Eigen::Matrix<std::complex<double>, DIM, 1> u_k;
  if (DIM == 2) {
    size_t i = 0;
    for (k[0] = 0; k[0] < grid.N[0]; k[0]++) {
      for (k[1] = 0; k[1] < grid.N[1]; k[1]++) {
        grid.modal_stiffness(k, K_k);
        u_k(0) = u_hat.cpp_data[i];
        u_k(1) = u_hat.cpp_data[i + ncells];
        auto Ku_k = K_k * u_k;
        Ku_hat.cpp_data[i] = Ku_k(0);
        Ku_hat.cpp_data[i + ncells] = Ku_k(1);
        i++;
      }
    }
  }
  for (size_t i = 0; i < DIM; i++) fftw_execute(idft_Ku[i]);
  double correction = 1.0;
  for (size_t i = 0; i < DIM; i++) correction *= grid.L[i] / grid.N[i];
  correction /= ncells;
  // The following correction is due to the fact that
  //
  // 1. FFTW's backward Fourier transform returns the inverse DFT, scaled by the
  //    number of cells (see FFTW's FAQ, 3.10).
  // 2. The matrix \hat{K}_k^N in [Bri17] is a scaled modal stiffness, as shown
  //    by Eq. (45), where this matrix must be scaled by the cell volume to get
  //    the potential energy.
  for (size_t i = 0; i < ndofs; i++) {
    Ku.cpp_data[i] *= correction;
  }
}

template <size_t DIM>
void StiffnessMatrixFactory<DIM>::run(Eigen::MatrixXcd &K) {
  for (size_t i = 0; i < ndofs; i++) {
    u.cpp_data[i] = 0;
  }
  for (size_t j = 0; j < ndofs; j++) {
    u.cpp_data[j] = 1;
    compute_Ku();
    for (size_t i = 0; i < ndofs; i++) {
      K(i, j) = Ku.cpp_data[i];
    }
    u.cpp_data[j] = 0;
  }
}

int main() {
  const size_t dim = 2;
  const double mu = 1.0;
  const double nu = 0.3;
  size_t N[] = {3, 4};
  double L[] = {3. * 1.1, 4. * 1.2};
  StiffnessMatrixFactory<dim> factory{mu, nu, L, N};
  Eigen::MatrixXcd K_act{factory.ndofs, factory.ndofs};
  factory.run(K_act);

  CartesianGrid<dim> grid{mu, nu, L, N};

  const size_t num_dofs = grid.num_cells * dim;
  const size_t num_dofs_per_cell = grid.num_nodes_per_cell * dim;
  Eigen::MatrixXd Ke{num_dofs_per_cell, num_dofs_per_cell};
  // This is a copy-paste from Maxima
  Ke << 1.578282828282828, 0.3308080808080808, -1.119949494949495,
      -0.7891414141414141, 0.625, -0.125, 0.125, -0.625, 0.3308080808080808,
      1.578282828282828, -0.7891414141414141, -1.119949494949495, 0.125, -0.625,
      0.625, -0.125, -1.119949494949495, -0.7891414141414141, 1.578282828282828,
      0.3308080808080808, -0.125, 0.625, -0.625, 0.125, -0.7891414141414141,
      -1.119949494949495, 0.3308080808080808, 1.578282828282828, -0.625, 0.125,
      -0.125, 0.625, 0.625, 0.125, -0.125, -0.625, 1.433080808080808,
      -0.8876262626262627, 0.1710858585858586, -0.7165404040404041, -0.125,
      -0.625, 0.625, 0.125, -0.8876262626262627, 1.433080808080808,
      -0.7165404040404041, 0.1710858585858586, 0.125, 0.625, -0.625, -0.125,
      0.1710858585858586, -0.7165404040404041, 1.433080808080808,
      -0.8876262626262627, -0.625, -0.125, 0.125, 0.625, -0.7165404040404041,
      0.1710858585858586, -0.8876262626262627, 1.433080808080808;

  Eigen::MatrixXd K_exp{num_dofs, num_dofs};
  for (size_t i = 0; i < num_dofs; i++) {
    for (size_t j = 0; j < num_dofs; j++) {
      K_exp(i, j) = 0;
    }
  }
  size_t nodes[grid.num_nodes_per_cell];
  for (size_t cell = 0; cell < grid.num_cells; cell++) {
    grid.get_cell_nodes(cell, nodes);
    for (size_t ie = 0; ie < num_dofs_per_cell; ie++) {
      size_t i = nodes[ie % grid.num_nodes_per_cell] +
                 grid.num_cells * (ie / grid.num_nodes_per_cell);
      for (size_t je = 0; je < num_dofs_per_cell; je++) {
        size_t j = nodes[je % grid.num_nodes_per_cell] +
                   grid.num_cells * (je / grid.num_nodes_per_cell);
        K_exp(i, j) += Ke(ie, je);
      }
    }
  }

  double atol = 1e-15;
  double rtol = 1e-15;

  for (size_t i = 0; i < num_dofs; i++) {
    for (size_t j = 0; j < num_dofs; j++) {
      double act = std::real(K_act(i, j));
      if (fabs(std::imag(act)) > atol) {
        throw std::runtime_error("");
      }
      double exp = K_exp(i, j);
      double tol = rtol * fabs(exp) + atol;
      double err = fabs(act - exp);
      if (err > tol) {
        throw std::runtime_error("");
      }
    }
  }
}
