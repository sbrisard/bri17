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

template <int DIM>
class CartesianGrid {
 public:
  double L[DIM];
  size_t N[DIM];

  CartesianGrid(double L[], size_t N[]) {
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
      double const *k, Eigen::Matrix<std::complex<double>, DIM, 1> &B);
  void modal_stiffness(double const *k, double mu, double nu,
                       Eigen::Matrix<std::complex<double>, DIM, DIM> &K);
};

template <int DIM>
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

template <int DIM>
void CartesianGrid<DIM>::modal_strain_displacement(
    double const *k, Eigen::Matrix<std::complex<double>, DIM, 1> &B) {
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

template <int DIM>
void CartesianGrid<DIM>::modal_stiffness(
    double const *k, double mu, double nu,
    Eigen::Matrix<std::complex<double>, DIM, DIM> &K) {
  // {phi, chi, psi}[i] = {φ, χ, psi}(z_i) in the notation of [Bri17]
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
    K(1, 0) = K[1];

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

int main() {
  size_t N2[] = {32, 64};
  double L2[] = {0.5, 1.};
  CartesianGrid<2> grid2{L2, N2};
  std::cout << grid2 << std::endl;

  size_t N3[] = {32, 64, 128};
  double L3[] = {0.5, 1., 2.};
  CartesianGrid<3> grid3{L3, N3};
  std::cout << grid3 << std::endl;

  size_t N = 16;
  FFTWComplexBuffer in{N};
  in.cpp_data[0] = 1.;
  FFTWComplexBuffer out{N};
  auto p = fftw_plan_dft_1d(N, in.c_data, out.c_data, FFTW_FORWARD, FFTW_ESTIMATE);
  auto p_inv = fftw_plan_dft_1d(N, out.c_data, in.c_data, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  for (size_t i = 0; i < N; i++) {
    std::cout << out.cpp_data[i] << ", ";
  }
  std::cout << std::endl;
  fftw_execute(p_inv);
  for (size_t i = 0; i < N; i++) {
    std::cout << in.cpp_data[i] << ", ";
  }
  std::cout << std::endl;
  fftw_destroy_plan(p);
}
