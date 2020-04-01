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
 public:
  double mu;
  double nu;
  double L[DIM];
  size_t N[DIM];

  CartesianGrid(double mu, double nu, double L[], size_t N[]) {
    this->mu = mu;
    this->nu = nu;
    if ((DIM < 2) || (DIM > 3)) {
      throw std::domain_error(
          "DIM template integer parameter must be 2 or 3 (got " +
          std::to_string(DIM) + ")");
    }
    for (size_t i = 0; i < DIM; i++) {
      this->L[i] = L[i];
      this->N[i] = N[i];
    }
  }

  ~CartesianGrid() {}

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

  const double scaling = 1. / (1. - 2. * nu);
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

Eigen::Matrix<double, 8, 8> &element_stiffness_matrix_2d(double a, double b,
                                                         double mu, double nu) {
  Eigen::Matrix<double, 8, 8> K_I, K_II;
  double k11, k12, k13, k14, k55, k56, k57, k58;
  double b3a = b / 3. / a;
  double b6a = b / 6. / a;
  double a3b = a / 3. / b;
  double a6b = a / 6. / b;
  k11 = b / 3. / a;
  k12 = b / 6. / a;
  k13 = -b / 3. / a;
  k14 = -b / 6. / a;
  k55 = a / 3. / b;
  k56 = -a / 3. / b;
  k57 = a / 6. / b;
  k58 = -a / 6. / b;
  // clang-format off
  K_I <<  k11,  k12,  k13,  k14,  .25, -.25,  .25, -.25,
          k12,  k11,  k14,  k13,  .25, -.25,  .25, -.25,
          k13,  k14,  k11,  k12, -.25,  .25, -.25,  .25,
          k14,  k13,  k12,  k11, -.25,  .25, -.25,  .25,
          .25,  .25, -.25, -.25,  k55,  k56,  k57,  k58,
         -.25, -.25,  .25,  .25,  k56,  k55,  k58,  k57,
          .25,  .25, -.25, -.25,  k57,  k58,  k55,  k56,
         -.25, -.25,  .25,  .25,  k58,  k57,  k56,  k55;
  // clang-format on
  k11 = 2. * b / 3. * a + a / 3. / b;
  k12 = b / 3. / a - a / 3. / b;
  k13 = a / 6. / b - 2. * b / 3. / a;
  k14 = -b / 3. / a - a / 6. / b;
  k55 = b / 3. / a + 2. * a / 3. / b;
  k56 = b / 6. / a - 2. * a / 3. / b;
  k57 = a / 3. / b - b / 3. / a;
  k58 = -b / 6. / a - a / 3. / b;
  // clang-format off
  K_II <<  k11,  k12,  k13,  k14,  .25,  .25, -.25, -.25,
           k12,  k11,  k14,  k13, -.25, -.25,  .25,  .25,
           k13,  k14,  k11,  k12,  .25,  .25, -.25, -.25,
           k14,  k13,  k12,  k11, -.25, -.25,  .25,  .25,
           .25, -.25,  .25, -.25,  k55,  k56,  k57,  k58,
           .25, -.25,  .25, -.25,  k56,  k55,  k58,  k57,
          -.25,  .25, -.25,  .25,  k57,  k58,  k55,  k56,
          -.25,  .25, -.25,  .25,  k58,  k57,  k56,  k55;
  // clang-format on
}

int main() {
  const size_t dim = 2;
  const double mu = 1.0;
  const double nu = 0.3;
  double L[] = {1.1, 1.2};
  size_t N[] = {3, 4};
  StiffnessMatrixFactory<dim> factory{mu, nu, L, N};
  Eigen::MatrixXcd K_act{factory.ndofs, factory.ndofs};
  factory.run(K_act);
  std::cout << K_act << std::endl;
}
