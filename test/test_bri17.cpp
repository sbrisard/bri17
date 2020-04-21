#include "bri17/bri17.hpp"
#include "catch2/catch.hpp"

#include <fftw3.h>

void assert_equal(const Eigen::MatrixXd &expected,
                  const Eigen::MatrixXd &actual, double rtol, double atol) {
  REQUIRE(expected.rows() == actual.rows());
  REQUIRE(expected.cols() == actual.cols());
  for (size_t i = 0; i < expected.rows(); i++) {
    for (size_t j = 0; j < expected.cols(); j++) {
      double e_ij = expected(i, j);
      double a_ij = actual(i, j);
      double tol = rtol * fabs(e_ij) + atol;
      double err = fabs(a_ij - e_ij);
      REQUIRE(err <= tol);
    }
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
    for (size_t i = 0; i < DIM; i++) {
      size_t offset = i * hooke.grid.num_cells;
      dft_u[i] =
          fftw_plan_dft(DIM, N_, u.c_data + offset, u_hat.c_data + offset,
                        FFTW_FORWARD, FFTW_ESTIMATE);
      idft_Ku[i] =
          fftw_plan_dft(DIM, N_, Ku_hat.c_data + offset, Ku.c_data + offset,
                        FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }

  void run(Eigen::MatrixXd &K) {
    for (size_t i = 0; i < ndofs; i++) {
      u.cpp_data[i] = 0;
    }
    for (size_t j = 0; j < ndofs; j++) {
      u.cpp_data[j] = 1;
      compute_Ku();
      for (size_t i = 0; i < ndofs; i++) {
        const auto K_ij = Ku.cpp_data[i];
        if (fabs(std::imag(K_ij)) > 1e-14) {
          throw std::runtime_error(std::to_string(std::imag(K_ij)));
        }
        K(i, j) = std::real(K_ij);
      }
      u.cpp_data[j] = 0;
    }
  };
};

template <size_t DIM>
void assemble_expected_stiffness_matrix(const bri17::CartesianGrid<DIM> &grid,
                                        const Eigen::MatrixXd &Ke,
                                        Eigen::MatrixXd &K) {
  // TODO Check dimensions of Ke and K.
  const size_t num_dofs_per_cell = grid.num_nodes_per_cell * DIM;
  const size_t num_dofs = grid.num_cells * DIM;
  K.setZero();
  size_t cell_nodes[grid.num_nodes_per_cell];
  for (size_t cell = 0; cell < grid.num_cells; cell++) {
    grid.get_cell_nodes(cell, cell_nodes);
    for (size_t ie = 0; ie < num_dofs_per_cell; ie++) {
      size_t i = cell_nodes[ie % grid.num_nodes_per_cell] +
                 grid.num_cells * (ie / grid.num_nodes_per_cell);
      for (size_t je = 0; je < num_dofs_per_cell; je++) {
        size_t j = cell_nodes[je % grid.num_nodes_per_cell] +
                   grid.num_cells * (je / grid.num_nodes_per_cell);
        K(i, j) += Ke(ie, je);
      }
    }
  }
}

TEST_CASE("2D stiffness matrix") {
  const size_t dim = 2;
  const double mu = 5.6;
  const double nu = 0.3;
  size_t N[] = {3, 4};
  double L[] = {3. * 1.1, 4. * 1.2};
  bri17::CartesianGrid<dim> grid{N, L};

  const size_t num_dofs_per_cell = grid.num_nodes_per_cell * dim;
  Eigen::MatrixXd Ke{num_dofs_per_cell, num_dofs_per_cell};
  // This is a copy-paste from Maxima
  Ke << 8.83838383838384, 1.852525252525252, -6.271717171717172,
      -4.41919191919192, 3.5, -0.7, 0.7, -3.5, 1.852525252525252,
      8.83838383838384, -4.41919191919192, -6.271717171717172, 0.7, -3.5, 3.5,
      -0.7, -6.271717171717172, -4.41919191919192, 8.83838383838384,
      1.852525252525252, -0.7, 3.5, -3.5, 0.7, -4.41919191919192,
      -6.271717171717172, 1.852525252525252, 8.83838383838384, -3.5, 0.7, -0.7,
      3.5, 3.5, 0.7, -0.7, -3.5, 8.025252525252526, -4.97070707070707,
      0.9580808080808081, -4.012626262626263, -0.7, -3.5, 3.5, 0.7,
      -4.97070707070707, 8.025252525252526, -4.012626262626263,
      0.9580808080808081, 0.7, 3.5, -3.5, -0.7, 0.9580808080808081,
      -4.012626262626263, 8.025252525252526, -4.97070707070707, -3.5, -0.7, 0.7,
      3.5, -4.012626262626263, 0.9580808080808081, -4.97070707070707,
      8.025252525252526;

  const size_t num_dofs = grid.num_cells * dim;
  Eigen::MatrixXd K_exp{num_dofs, num_dofs};
  assemble_expected_stiffness_matrix(grid, Ke, K_exp);

  bri17::Hooke<dim> hooke{mu, nu, grid};
  StiffnessMatrixFactory<dim> factory{hooke};
  Eigen::MatrixXd K_act{num_dofs, num_dofs};
  factory.run(K_act);

  assert_equal(K_exp, K_act, 1e-15, 1e-14);
}
