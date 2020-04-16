#include "bri17/bri17.hpp"

#include <fftw3.h>

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
 private:
  const size_t ndofs;
  const bri17::Hooke<DIM> hooke;
  const FFTWComplexBuffer u;
  const FFTWComplexBuffer u_hat;
  const FFTWComplexBuffer Ku;
  const FFTWComplexBuffer Ku_hat;
  // TODO These members should be const
  fftw_plan dft_u[DIM];
  fftw_plan idft_Ku[DIM];

  void compute_Ku() {
    for (size_t i = 0; i < DIM; i++) fftw_execute(dft_u[i]);
    size_t k[DIM] = {0};
    Eigen::Matrix<std::complex<double>, DIM, DIM> K_k;
    Eigen::Matrix<std::complex<double>, DIM, 1> u_k;
    if (DIM == 2) {
      size_t i = 0;
      for (k[0] = 0; k[0] < hooke.grid.N[0]; k[0]++) {
        for (k[1] = 0; k[1] < hooke.grid.N[1]; k[1]++) {
          hooke.modal_stiffness(k, K_k);
          u_k(0) = u_hat.cpp_data[i];
          u_k(1) = u_hat.cpp_data[i + hooke.grid.num_cells];
          auto Ku_k = K_k * u_k;
          Ku_hat.cpp_data[i] = Ku_k(0);
          Ku_hat.cpp_data[i + hooke.grid.num_cells] = Ku_k(1);
          i++;
        }
      }
    }
    for (size_t i = 0; i < DIM; i++) fftw_execute(idft_Ku[i]);
    double correction = 1.0;
    for (size_t i = 0; i < DIM; i++)
      correction *= hooke.grid.L[i] / hooke.grid.N[i];
    correction /= hooke.grid.num_cells;
    // The following correction is due to the fact that
    //
    // 1. FFTW's backward Fourier transform returns the inverse DFT, scaled by
    //    the number of cells (see FFTW's FAQ, 3.10).
    // 2. The matrix \hat{K}_k^N in [Bri17] is a scaled modal stiffness, as
    //    shown by Eq. (45), where this matrix must be scaled by the cell volume
    //    to get the potential energy.
    for (size_t i = 0; i < ndofs; i++) {
      Ku.cpp_data[i] *= correction;
    }
  };

 public:
  StiffnessMatrixFactory(bri17::Hooke<DIM> hooke)
      : ndofs{DIM * hooke.grid.num_cells},
        hooke{hooke},
        u{ndofs},
        u_hat{ndofs},
        Ku{ndofs},
        Ku_hat{ndofs} {
    int N_[DIM];
    for (size_t i = 0; i < DIM; i++) N_[i] = hooke.grid.N[i];
    for (size_t k = 0; k < DIM; k++) {
      size_t offset = k * hooke.grid.num_cells;
      dft_u[k] =
          fftw_plan_dft(DIM, N_, u.c_data + offset, u_hat.c_data + offset,
                        FFTW_FORWARD, FFTW_ESTIMATE);
      idft_Ku[k] =
          fftw_plan_dft(DIM, N_, Ku_hat.c_data + offset, Ku.c_data + offset,
                        FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }

  void run(Eigen::MatrixXcd &K) {
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
  };
};

int main() {
  const size_t dim = 2;
  const double mu = 1.0;
  const double nu = 0.3;
  size_t N[] = {3, 4};
  double L[] = {3. * 1.1, 4. * 1.2};
  bri17::CartesianGrid<dim> grid{N, L};
  bri17::Hooke<dim> hooke{mu, nu, grid};
  const size_t num_dofs = grid.num_cells * dim;
  const size_t num_dofs_per_cell = grid.num_nodes_per_cell * dim;
  StiffnessMatrixFactory<dim> factory{hooke};
  Eigen::MatrixXcd K_act{num_dofs, num_dofs};
  factory.run(K_act);

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
