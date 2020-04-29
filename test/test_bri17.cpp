#include <sstream>
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
      CHECKED_ELSE(err <= tol) {
        std::ostringstream msg;
        msg << "[" << i << ", " << j << "]: expected = " << e_ij
            << ", actual = " << a_ij;
        FAIL(msg.str());
      }
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
  const size_t num_dofs;
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
    if constexpr (DIM == 2) {
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
    } else if constexpr (DIM == 3) {
      // TODO: the two cases should be merged
      size_t i = 0;
      for (k[0] = 0; k[0] < hooke.grid.N[0]; k[0]++) {
        for (k[1] = 0; k[1] < hooke.grid.N[1]; k[1]++) {
          for (k[2] = 0; k[2] < hooke.grid.N[2]; k[2]++) {
            hooke.modal_stiffness(k, K_k);
            u_k(0) = u_hat.cpp_data[i];
            u_k(1) = u_hat.cpp_data[i + hooke.grid.num_cells];
            u_k(2) = u_hat.cpp_data[i + 2 * hooke.grid.num_cells];
            auto Ku_k = K_k * u_k;
            Ku_hat.cpp_data[i] = Ku_k(0);
            Ku_hat.cpp_data[i + hooke.grid.num_cells] = Ku_k(1);
            Ku_hat.cpp_data[i + 2 * hooke.grid.num_cells] = Ku_k(2);
            i++;
          }
        }
      }
    }
    for (size_t i = 0; i < DIM; i++) fftw_execute(idft_Ku[i]);
    double correction = 1.0 / hooke.grid.num_cells;
    // The following correction is due to the fact that FFTW's backward Fourier
    // transform returns the inverse DFT, scaled by the number of cells (see
    // FFTW's FAQ, 3.10).
    for (size_t i = 0; i < num_dofs; i++) {
      Ku.cpp_data[i] *= correction;
    }
  };

 public:
  StiffnessMatrixFactory(bri17::Hooke<DIM> hooke)
      : num_dofs{DIM * hooke.grid.num_cells},
        hooke{hooke},
        u{num_dofs},
        u_hat{num_dofs},
        Ku{num_dofs},
        Ku_hat{num_dofs} {
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

  Eigen::MatrixXd run() {
    Eigen::MatrixXd K{num_dofs, num_dofs};
    for (size_t i = 0; i < num_dofs; i++) {
      u.cpp_data[i] = 0;
    }
    for (size_t j = 0; j < num_dofs; j++) {
      u.cpp_data[j] = 1;
      compute_Ku();
      for (size_t i = 0; i < num_dofs; i++) {
        const auto K_ij = Ku.cpp_data[i];
        if (fabs(std::imag(K_ij)) > 1e-14) {
          std::ostringstream msg;
          msg << "imag(K[" << i << ", " << j << "]) = " << std::imag(K_ij);
          throw std::runtime_error(msg.str());
        }
        K(i, j) = std::real(K_ij);
      }
      u.cpp_data[j] = 0;
    }
    return K;
  }
};

template <size_t DIM>
Eigen::MatrixXd assemble_expected_stiffness_matrix(
    const bri17::CartesianGrid<DIM> &grid, const Eigen::MatrixXd &Ke) {
  // TODO Check dimensions of Ke and K.
  const size_t num_dofs_per_cell = grid.num_nodes_per_cell * DIM;
  const size_t num_dofs = grid.num_cells * DIM;
  Eigen::MatrixXd K{num_dofs, num_dofs};
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
  return K;
}

template <size_t DIM>
constexpr size_t num_strain_components() {
  return (DIM * (DIM + 1)) / 2;
}

template <size_t DIM>
class StrainDisplacementMatrixFactory {
 private:
  const size_t num_dofs;
  const bri17::Hooke<DIM> hooke;
  const FFTWComplexBuffer u;
  const FFTWComplexBuffer u_hat;
  const FFTWComplexBuffer Bu;
  const FFTWComplexBuffer Bu_hat;
  // TODO These members should be const
  fftw_plan dft_u[DIM];
  fftw_plan idft_Bu[num_strain_components<DIM>()];

  void compute_Bu() {
    for (size_t i = 0; i < DIM; i++) fftw_execute(dft_u[i]);
    size_t k[DIM] = {0};
    Eigen::Matrix<std::complex<double>, num_strain_components<DIM>(), DIM> B_k;
    Eigen::Matrix<std::complex<double>, DIM, 1> u_k;
    if constexpr (DIM == 2) {
      size_t i = 0;
      for (k[0] = 0; k[0] < hooke.grid.N[0]; k[0]++) {
        for (k[1] = 0; k[1] < hooke.grid.N[1]; k[1]++) {
          hooke.modal_strain_displacement(k, B_k);
          u_k(0) = u_hat.cpp_data[i];
          u_k(1) = u_hat.cpp_data[i + hooke.grid.num_cells];
          auto Bu_k = B_k * u_k;
          for (size_t j = 0; j < num_strain_components<DIM>(); j++) {
            Bu_hat.cpp_data[i + j * hooke.grid.num_cells] = Bu_k(j);
          }
          i++;
        }
      }
    } else if constexpr (DIM == 3) {
      // TODO: the two cases should be merged
      size_t i = 0;
      for (k[0] = 0; k[0] < hooke.grid.N[0]; k[0]++) {
        for (k[1] = 0; k[1] < hooke.grid.N[1]; k[1]++) {
          for (k[2] = 0; k[2] < hooke.grid.N[2]; k[2]++) {
            hooke.modal_strain_displacement(k, B_k);
            u_k(0) = u_hat.cpp_data[i];
            u_k(1) = u_hat.cpp_data[i + hooke.grid.num_cells];
            u_k(2) = u_hat.cpp_data[i + 2 * hooke.grid.num_cells];
            auto Bu_k = B_k * u_k;
            for (size_t j = 0; j < num_strain_components<DIM>(); j++) {
              Bu_hat.cpp_data[i + j * hooke.grid.num_cells] = Bu_k(j);
            }
            i++;
          }
        }
      }
    }
    for (size_t i = 0; i < num_strain_components<DIM>(); i++)
      fftw_execute(idft_Bu[i]);
    double correction = 1.0 / hooke.grid.num_cells;
    // The following correction is due to the fact that FFTW's backward Fourier
    // transform returns the inverse DFT, scaled by the number of cells (see
    // FFTW's FAQ, 3.10).
    for (size_t i = 0; i < num_strain_components<DIM>() * hooke.grid.num_cells;
         i++) {
      Bu.cpp_data[i] *= correction;
    }
  };

 public:
  StrainDisplacementMatrixFactory(bri17::Hooke<DIM> hooke)
      : num_dofs{DIM * hooke.grid.num_cells},
        hooke{hooke},
        u{num_dofs},
        u_hat{num_dofs},
        Bu{num_strain_components<DIM>() * hooke.grid.num_cells},
        Bu_hat{num_strain_components<DIM>() * hooke.grid.num_cells} {
    int N_[DIM];
    for (size_t i = 0; i < DIM; i++) N_[i] = hooke.grid.N[i];
    for (size_t i = 0; i < DIM; i++) {
      size_t offset = i * hooke.grid.num_cells;
      dft_u[i] =
          fftw_plan_dft(DIM, N_, u.c_data + offset, u_hat.c_data + offset,
                        FFTW_FORWARD, FFTW_ESTIMATE);
    }
    for (size_t i = 0; i < num_strain_components<DIM>(); i++) {
      size_t offset = i * hooke.grid.num_cells;
      idft_Bu[i] =
          fftw_plan_dft(DIM, N_, Bu_hat.c_data + offset, Bu.c_data + offset,
                        FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }

  Eigen::MatrixXd run() {
    const size_t num_rows = num_strain_components<DIM>() * hooke.grid.num_cells;
    Eigen::MatrixXd B{num_rows, num_dofs};
    for (size_t i = 0; i < num_dofs; i++) {
      u.cpp_data[i] = 0;
    }
    for (size_t j = 0; j < num_dofs; j++) {
      u.cpp_data[j] = 1;
      compute_Bu();
      for (size_t i = 0; i < num_rows; i++) {
        const auto B_ij = Bu.cpp_data[i];
        if (fabs(std::imag(B_ij)) > 1e-14) {
          std::ostringstream msg;
          msg << "imag(B[" << i << ", " << j << "]) = " << std::imag(B_ij);
          throw std::runtime_error(msg.str());
        }
        B(i, j) = std::real(B_ij);
      }
      u.cpp_data[j] = 0;
    }
    return B;
  }
};

template <size_t DIM>
Eigen::MatrixXd assemble_expected_strain_displacement_matrix(
    const bri17::CartesianGrid<DIM> &grid, const Eigen::MatrixXd &Be) {
  const size_t num_strain_components = (DIM * (DIM + 1)) / 2;
  const size_t num_rows = grid.num_cells * num_strain_components;
  const size_t num_dofs_per_cell = grid.num_nodes_per_cell * DIM;
  const size_t num_dofs = grid.num_cells * DIM;
  Eigen::MatrixXd B{num_rows, num_dofs};
  B.setZero();
  size_t cell_nodes[grid.num_nodes_per_cell];
  for (size_t cell = 0; cell < grid.num_cells; cell++) {
    grid.get_cell_nodes(cell, cell_nodes);
    for (size_t ie = 0; ie < num_strain_components; ie++) {
      size_t i = cell * num_strain_components + ie;
      for (size_t je = 0; je < num_dofs_per_cell; je++) {
        size_t j = cell_nodes[je % grid.num_nodes_per_cell] +
                   grid.num_cells * (je / grid.num_nodes_per_cell);
        B(i, j) += Be(ie, je);
      }
    }
  }
  return B;
}

TEST_CASE("Global assembly tests") {
  const size_t max_dim = 3;
  const double mu = 5.6;
  const double nu = 0.3;
  // Arrays N and L are oversized for dim == 2, but that does not
  // matter
  size_t N[] = {3, 4, 5};
  double h[] = {1.1, 1.2, 1.3};
  double L[max_dim];
  for (size_t i = 0; i < max_dim; i++) {
    L[i] = N[i] * h[i];
  }

  SECTION("2D stiffness matrix") {
    const size_t dim = 2;
    bri17::CartesianGrid<dim> grid{N, L};
    bri17::Hooke<dim> hooke{mu, nu, grid};
    StiffnessMatrixFactory<dim> factory{hooke};

    const size_t num_dofs_per_cell = grid.num_nodes_per_cell * dim;
    Eigen::MatrixXd Ke{num_dofs_per_cell, num_dofs_per_cell};
    // This is a copy-paste from Maxima
    Ke << 8.83838383838384, 1.852525252525252, -6.271717171717172,
        -4.41919191919192, 3.5, -0.7, 0.7, -3.5, 1.852525252525252,
        8.83838383838384, -4.41919191919192, -6.271717171717172, 0.7, -3.5, 3.5,
        -0.7, -6.271717171717172, -4.41919191919192, 8.83838383838384,
        1.852525252525252, -0.7, 3.5, -3.5, 0.7, -4.41919191919192,
        -6.271717171717172, 1.852525252525252, 8.83838383838384, -3.5, 0.7,
        -0.7, 3.5, 3.5, 0.7, -0.7, -3.5, 8.025252525252526, -4.97070707070707,
        0.9580808080808081, -4.012626262626263, -0.7, -3.5, 3.5, 0.7,
        -4.97070707070707, 8.025252525252526, -4.012626262626263,
        0.9580808080808081, 0.7, 3.5, -3.5, -0.7, 0.9580808080808081,
        -4.012626262626263, 8.025252525252526, -4.97070707070707, -3.5, -0.7,
        0.7, 3.5, -4.012626262626263, 0.9580808080808081, -4.97070707070707,
        8.025252525252526;

    auto K_exp = assemble_expected_stiffness_matrix(grid, Ke);
    auto K_act = factory.run();
    assert_equal(K_exp, K_act, 1e-15, 1e-14);
  }

  SECTION("3D stiffness matrix") {
    const size_t dim = 3;
    bri17::CartesianGrid<dim> grid{N, L};
    bri17::Hooke<dim> hooke{mu, nu, grid};
    StiffnessMatrixFactory<dim> factory{hooke};

    const size_t num_dofs_per_cell = grid.num_nodes_per_cell * dim;
    Eigen::MatrixXd Ke{num_dofs_per_cell, num_dofs_per_cell};
    // This is a copy-paste from Maxima
    Ke << 4.461761201761202, 1.283188293188293, 1.118658378658379,
        0.08548303548303549, -2.401846671846672, -1.67476948976949,
        -1.757034447034447, -1.1154403004403, 1.516666666666667,
        0.7583333333333333, -0.3033333333333333, -0.1516666666666667,
        0.3033333333333333, 0.1516666666666667, -1.516666666666667,
        -0.7583333333333333, 1.4, -0.28, 0.7, -0.14, 0.28, -1.4, 0.14, -0.7,
        1.283188293188293, 4.461761201761202, 0.08548303548303549,
        1.118658378658379, -1.67476948976949, -2.401846671846672,
        -1.1154403004403, -1.757034447034447, 0.7583333333333333,
        1.516666666666667, -0.1516666666666667, -0.3033333333333333,
        0.1516666666666667, 0.3033333333333333, -0.7583333333333333,
        -1.516666666666667, 0.28, -1.4, 0.14, -0.7, 1.4, -0.28, 0.7, -0.14,
        1.118658378658379, 0.08548303548303549, 4.461761201761202,
        1.283188293188293, -1.757034447034447, -1.1154403004403,
        -2.401846671846672, -1.67476948976949, 0.3033333333333333,
        0.1516666666666667, -1.516666666666667, -0.7583333333333333,
        1.516666666666667, 0.7583333333333333, -0.3033333333333333,
        -0.1516666666666667, 0.7, -0.14, 1.4, -0.28, 0.14, -0.7, 0.28, -1.4,
        0.08548303548303549, 1.118658378658379, 1.283188293188293,
        4.461761201761202, -1.1154403004403, -1.757034447034447,
        -1.67476948976949, -2.401846671846672, 0.1516666666666667,
        0.3033333333333333, -0.7583333333333333, -1.516666666666667,
        0.7583333333333333, 1.516666666666667, -0.1516666666666667,
        -0.3033333333333333, 0.14, -0.7, 0.28, -1.4, 0.7, -0.14, 1.4, -0.28,
        -2.401846671846672, -1.67476948976949, -1.757034447034447,
        -1.1154403004403, 4.461761201761202, 1.283188293188293,
        1.118658378658379, 0.08548303548303549, -0.3033333333333333,
        -0.1516666666666667, 1.516666666666667, 0.7583333333333333,
        -1.516666666666667, -0.7583333333333333, 0.3033333333333333,
        0.1516666666666667, -0.28, 1.4, -0.14, 0.7, -1.4, 0.28, -0.7, 0.14,
        -1.67476948976949, -2.401846671846672, -1.1154403004403,
        -1.757034447034447, 1.283188293188293, 4.461761201761202,
        0.08548303548303549, 1.118658378658379, -0.1516666666666667,
        -0.3033333333333333, 0.7583333333333333, 1.516666666666667,
        -0.7583333333333333, -1.516666666666667, 0.1516666666666667,
        0.3033333333333333, -1.4, 0.28, -0.7, 0.14, -0.28, 1.4, -0.14, 0.7,
        -1.757034447034447, -1.1154403004403, -2.401846671846672,
        -1.67476948976949, 1.118658378658379, 0.08548303548303549,
        4.461761201761202, 1.283188293188293, -1.516666666666667,
        -0.7583333333333333, 0.3033333333333333, 0.1516666666666667,
        -0.3033333333333333, -0.1516666666666667, 1.516666666666667,
        0.7583333333333333, -0.14, 0.7, -0.28, 1.4, -0.7, 0.14, -1.4, 0.28,
        -1.1154403004403, -1.757034447034447, -1.67476948976949,
        -2.401846671846672, 0.08548303548303549, 1.118658378658379,
        1.283188293188293, 4.461761201761202, -0.7583333333333333,
        -1.516666666666667, 0.1516666666666667, 0.3033333333333333,
        -0.1516666666666667, -0.3033333333333333, 0.7583333333333333,
        1.516666666666667, -0.7, 0.14, -1.4, 0.28, -0.14, 0.7, -0.28, 1.4,
        1.516666666666667, 0.7583333333333333, 0.3033333333333333,
        0.1516666666666667, -0.3033333333333333, -0.1516666666666667,
        -1.516666666666667, -0.7583333333333333, 4.1094042994043,
        1.107009842009842, -1.838075628075628, -1.392883967883968,
        0.7310657860657861, -0.1083132608132608, -1.580855995855996,
        -1.027351074851075, 1.283333333333333, -0.2566666666666667,
        0.2566666666666667, -1.283333333333333, 0.6416666666666667,
        -0.1283333333333333, 0.1283333333333333, -0.6416666666666667,
        0.7583333333333333, 1.516666666666667, 0.1516666666666667,
        0.3033333333333333, -0.1516666666666667, -0.3033333333333333,
        -0.7583333333333333, -1.516666666666667, 1.107009842009842,
        4.1094042994043, -1.392883967883968, -1.838075628075628,
        -0.1083132608132608, 0.7310657860657861, -1.027351074851075,
        -1.580855995855996, 0.2566666666666667, -1.283333333333333,
        1.283333333333333, -0.2566666666666667, 0.1283333333333333,
        -0.6416666666666667, 0.6416666666666667, -0.1283333333333333,
        -0.3033333333333333, -0.1516666666666667, -1.516666666666667,
        -0.7583333333333333, 1.516666666666667, 0.7583333333333333,
        0.3033333333333333, 0.1516666666666667, -1.838075628075628,
        -1.392883967883968, 4.1094042994043, 1.107009842009842,
        -1.580855995855996, -1.027351074851075, 0.7310657860657861,
        -0.1083132608132608, -0.2566666666666667, 1.283333333333333,
        -1.283333333333333, 0.2566666666666667, -0.1283333333333333,
        0.6416666666666667, -0.6416666666666667, 0.1283333333333333,
        -0.1516666666666667, -0.3033333333333333, -0.7583333333333333,
        -1.516666666666667, 0.7583333333333333, 1.516666666666667,
        0.1516666666666667, 0.3033333333333333, -1.392883967883968,
        -1.838075628075628, 1.107009842009842, 4.1094042994043,
        -1.027351074851075, -1.580855995855996, -0.1083132608132608,
        0.7310657860657861, -1.283333333333333, 0.2566666666666667,
        -0.2566666666666667, 1.283333333333333, -0.6416666666666667,
        0.1283333333333333, -0.1283333333333333, 0.6416666666666667,
        0.3033333333333333, 0.1516666666666667, 1.516666666666667,
        0.7583333333333333, -1.516666666666667, -0.7583333333333333,
        -0.3033333333333333, -0.1516666666666667, 0.7310657860657861,
        -0.1083132608132608, -1.580855995855996, -1.027351074851075,
        4.1094042994043, 1.107009842009842, -1.838075628075628,
        -1.392883967883968, 0.6416666666666667, -0.1283333333333333,
        0.1283333333333333, -0.6416666666666667, 1.283333333333333,
        -0.2566666666666667, 0.2566666666666667, -1.283333333333333,
        0.1516666666666667, 0.3033333333333333, 0.7583333333333333,
        1.516666666666667, -0.7583333333333333, -1.516666666666667,
        -0.1516666666666667, -0.3033333333333333, -0.1083132608132608,
        0.7310657860657861, -1.027351074851075, -1.580855995855996,
        1.107009842009842, 4.1094042994043, -1.392883967883968,
        -1.838075628075628, 0.1283333333333333, -0.6416666666666667,
        0.6416666666666667, -0.1283333333333333, 0.2566666666666667,
        -1.283333333333333, 1.283333333333333, -0.2566666666666667,
        -1.516666666666667, -0.7583333333333333, -0.3033333333333333,
        -0.1516666666666667, 0.3033333333333333, 0.1516666666666667,
        1.516666666666667, 0.7583333333333333, -1.580855995855996,
        -1.027351074851075, 0.7310657860657861, -0.1083132608132608,
        -1.838075628075628, -1.392883967883968, 4.1094042994043,
        1.107009842009842, -0.1283333333333333, 0.6416666666666667,
        -0.6416666666666667, 0.1283333333333333, -0.2566666666666667,
        1.283333333333333, -1.283333333333333, 0.2566666666666667,
        -0.7583333333333333, -1.516666666666667, -0.1516666666666667,
        -0.3033333333333333, 0.1516666666666667, 0.3033333333333333,
        0.7583333333333333, 1.516666666666667, -1.027351074851075,
        -1.580855995855996, -0.1083132608132608, 0.7310657860657861,
        -1.392883967883968, -1.838075628075628, 1.107009842009842,
        4.1094042994043, -0.6416666666666667, 0.1283333333333333,
        -0.1283333333333333, 0.6416666666666667, -1.283333333333333,
        0.2566666666666667, -0.2566666666666667, 1.283333333333333, 1.4, 0.28,
        0.7, 0.14, -0.28, -1.4, -0.14, -0.7, 1.283333333333333,
        0.2566666666666667, -0.2566666666666667, -1.283333333333333,
        0.6416666666666667, 0.1283333333333333, -0.1283333333333333,
        -0.6416666666666667, 3.835187775187775, -1.399329189329189,
        0.8053716653716654, -1.255775705775706, 0.593957523957524,
        -1.361482776482776, -0.2591323491323491, -0.9587969437969438, -0.28,
        -1.4, -0.14, -0.7, 1.4, 0.28, 0.7, 0.14, -0.2566666666666667,
        -1.283333333333333, 1.283333333333333, 0.2566666666666667,
        -0.1283333333333333, -0.6416666666666667, 0.6416666666666667,
        0.1283333333333333, -1.399329189329189, 3.835187775187775,
        -1.255775705775706, 0.8053716653716654, -1.361482776482776,
        0.593957523957524, -0.9587969437969438, -0.2591323491323491, 0.7, 0.14,
        1.4, 0.28, -0.14, -0.7, -0.28, -1.4, 0.2566666666666667,
        1.283333333333333, -1.283333333333333, -0.2566666666666667,
        0.1283333333333333, 0.6416666666666667, -0.6416666666666667,
        -0.1283333333333333, 0.8053716653716654, -1.255775705775706,
        3.835187775187775, -1.399329189329189, -0.2591323491323491,
        -0.9587969437969438, 0.593957523957524, -1.361482776482776, -0.14, -0.7,
        -0.28, -1.4, 0.7, 0.14, 1.4, 0.28, -1.283333333333333,
        -0.2566666666666667, 0.2566666666666667, 1.283333333333333,
        -0.6416666666666667, -0.1283333333333333, 0.1283333333333333,
        0.6416666666666667, -1.255775705775706, 0.8053716653716654,
        -1.399329189329189, 3.835187775187775, -0.9587969437969438,
        -0.2591323491323491, -1.361482776482776, 0.593957523957524, 0.28, 1.4,
        0.14, 0.7, -1.4, -0.28, -0.7, -0.14, 0.6416666666666667,
        0.1283333333333333, -0.1283333333333333, -0.6416666666666667,
        1.283333333333333, 0.2566666666666667, -0.2566666666666667,
        -1.283333333333333, 0.593957523957524, -1.361482776482776,
        -0.2591323491323491, -0.9587969437969438, 3.835187775187775,
        -1.399329189329189, 0.8053716653716654, -1.255775705775706, -1.4, -0.28,
        -0.7, -0.14, 0.28, 1.4, 0.14, 0.7, -0.1283333333333333,
        -0.6416666666666667, 0.6416666666666667, 0.1283333333333333,
        -0.2566666666666667, -1.283333333333333, 1.283333333333333,
        0.2566666666666667, -1.361482776482776, 0.593957523957524,
        -0.9587969437969438, -0.2591323491323491, -1.399329189329189,
        3.835187775187775, -1.255775705775706, 0.8053716653716654, 0.14, 0.7,
        0.28, 1.4, -0.7, -0.14, -1.4, -0.28, 0.1283333333333333,
        0.6416666666666667, -0.6416666666666667, -0.1283333333333333,
        0.2566666666666667, 1.283333333333333, -1.283333333333333,
        -0.2566666666666667, -0.2591323491323491, -0.9587969437969438,
        0.593957523957524, -1.361482776482776, 0.8053716653716654,
        -1.255775705775706, 3.835187775187775, -1.399329189329189, -0.7, -0.14,
        -1.4, -0.28, 0.14, 0.7, 0.28, 1.4, -0.6416666666666667,
        -0.1283333333333333, 0.1283333333333333, 0.6416666666666667,
        -1.283333333333333, -0.2566666666666667, 0.2566666666666667,
        1.283333333333333, -0.9587969437969438, -0.2591323491323491,
        -1.361482776482776, 0.593957523957524, -1.255775705775706,
        0.8053716653716654, -1.399329189329189, 3.835187775187775;

    auto K_exp = assemble_expected_stiffness_matrix(grid, Ke);
    auto K_act = factory.run();
    assert_equal(K_exp, K_act, 1e-15, 1e-14);
  }

  SECTION("2D strain-displacement matrix") {
    const size_t dim = 2;
    bri17::CartesianGrid<dim> grid{N, L};
    bri17::Hooke<dim> hooke{mu, nu, grid};

    const size_t num_dofs_per_cell = grid.num_nodes_per_cell * dim;
    const size_t num_strain_components_per_cell = (dim * (dim + 1)) / 2;
    Eigen::MatrixXd Be{num_strain_components_per_cell, num_dofs_per_cell};
    // This is a copy-paste from Maxima
    Be << -0.6, -0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.55,
        0.55, -0.55, 0.55, -0.275, 0.275, -0.275, 0.275, -0.3, -0.3, 0.3, 0.3;
    auto B_exp = assemble_expected_strain_displacement_matrix(grid, Be);

    StrainDisplacementMatrixFactory<dim> factory{hooke};
    auto B_act = factory.run();

    assert_equal(B_exp, B_act, 1e-15, 1e-14);
  }
}
