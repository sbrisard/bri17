#include "bri17/bri17.h"
#include <math.h>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

const size_t MAX_DIM = 3;

template <int DIM>
class CartesianGrid {
 public:
  double *L;
  size_t *N;

  CartesianGrid(double L[], size_t N[]) {
    if ((DIM < 2) || (DIM > 3)) {
      throw std::domain_error(
          "DIM template integer parameter must be 2 or 3 (got " +
          std::to_string(DIM) + ")");
    }
    this->L = new double[DIM];
    this->N = new size_t[DIM];
    for (size_t i = 0; i < DIM; i++) {
      this->L[i] = L[i];
      this->N[i] = N[i];
    }
  }

  ~CartesianGrid() {
    delete[] L;
    delete[] N;
  }

  void modal_strain_displacement(double const *k, std::complex<double> *B);
  void modal_stiffness(double const *k, double mu, double nu,
                       std::complex<double> *K);
};

template <int DIM>
std::ostream &operator<<(std::ostream &os, const CartesianGrid<DIM> &grid) {
  os << "CartesianGrid<" << DIM << ">={L=[";
  // TODO Use range-based loops when templates are implemented.
  for (size_t i = 0; i < DIM; i++) {
    os << grid.L[i] << ",";
  }
  os << "],N=[";
  for (size_t i = 0; i < DIM; i++) {
    os << grid.N[i] << ",";
  }
  return os << "]}";
}

template <int DIM>
void CartesianGrid<DIM>::modal_strain_displacement(double const *k,
                                                   std::complex<double> *B) {
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
    B[0] = prefactor * h_inv[0] * s[0] * c[1];
    B[1] = prefactor * h_inv[1] * c[0] * s[1];
  } else if (DIM == 3) {
    B[0] = prefactor * h_inv[0] * s[0] * c[1] * c[2];
    B[1] = prefactor * h_inv[1] * c[0] * s[1] * c[2];
    B[2] = prefactor * h_inv[2] * c[0] * c[1] * s[2];
  } else {
    // This should never occur
    // TODO throw exception
  }
}

template <int DIM>
void CartesianGrid<DIM>::modal_stiffness(double const *k, double mu, double nu,
                                         std::complex<double> *K) {
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
    const double H0 = h_inv[0] * h_inv[0] * phi[0] * chi[1];
    K[0] = scaling * H0;
    K[1] = scaling * h_inv[0] * h_inv[1] * psi[0] * psi[1];
    const double H3 = h_inv[1] * h_inv[1] * chi[0] * phi[1];
    K[3] = scaling * H3;

    // Symmetrization
    K[2] = K[1];

    const double K_diag = mu * (H0 + H3);
    K[0] += K_diag;
    K[3] += K_diag;
  } else if (DIM == 3) {
    const double H0 = h_inv[0] * h_inv[0] * phi[0] * chi[1] * chi[2];
    K[0] = scaling * H0;
    K[1] = scaling * h_inv[0] * h_inv[1] * psi[0] * psi[1] * chi[2];
    K[2] = scaling * h_inv[0] * h_inv[2] * psi[0] * chi[1] * psi[2];
    const double H4 = h_inv[1] * h_inv[1] * chi[0] * phi[1] * chi[2];
    K[4] = scaling * H4;
    K[5] = scaling * h_inv[1] * h_inv[2] * chi[0] * psi[1] * psi[2];
    const double H8 = h_inv[2] * h_inv[2] * chi[0] * chi[1] * phi[2];
    K[8] = scaling * H8;

    K[3] = K[1];
    K[6] = K[2];
    K[7] = K[5];

    const double K_diag = mu * (H0 + H4 + H8);
    K[0] += K_diag;
    K[4] += K_diag;
    K[8] += K_diag;
  } else {
    // This should never occur
    // TODO throw exception
  }
}

int main() {
  size_t N2[] = {32, 64};
  double L2[] = {0.5, 1.};
  CartesianGrid<2> grid2{L2, N2};
  std::cout << grid2 << std::endl;

  size_t N3[] = {32, 64, 128};
  double L3[] = {0.5, 1., 2.};
  CartesianGrid<3> grid3{L3, N3};
  std::cout << grid3 << std::endl;

  CartesianGrid<4> grid4{L3, N3};
  std::cout << grid4 << std::endl;
}
