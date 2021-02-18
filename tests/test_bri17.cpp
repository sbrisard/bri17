#include <sstream>
#include "catch2/catch.hpp"

#include <fftw3.h>

#include "Eigen/Dense"
#include "bri17/bri17.hpp"

void assert_equal(const Eigen::MatrixXd &expected,
                  const Eigen::MatrixXd &actual, double rtol, double atol) {
  REQUIRE(expected.rows() == actual.rows());
  REQUIRE(expected.cols() == actual.cols());
  for (Eigen::Index i = 0; i < expected.rows(); i++) {
    for (Eigen::Index j = 0; j < expected.cols(); j++) {
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

  FFTWComplexBuffer(int n) {
    c_data = (fftw_complex *)fftw_malloc(n * sizeof(fftw_complex));
    cpp_data = reinterpret_cast<std::complex<double> *>(c_data);
  }

  ~FFTWComplexBuffer() { fftw_free(c_data); }
};

template <typename T, int DIM>
class StiffnessMatrixFactory {
 private:
  const int num_dofs;
  const bri17::Hooke<T, DIM> hooke;
  const FFTWComplexBuffer u;
  const FFTWComplexBuffer u_hat;
  const FFTWComplexBuffer Ku;
  const FFTWComplexBuffer Ku_hat;
  // TODO These members should be const
  fftw_plan dft_u[DIM];
  fftw_plan idft_Ku[DIM];

  void compute_Ku() {
    for (int i = 0; i < DIM; i++) fftw_execute(dft_u[i]);
    int k[DIM] = {0};
    Eigen::Matrix<std::complex<double>, DIM, DIM> K_k;
    Eigen::Matrix<std::complex<double>, DIM, 1> u_k;
    if constexpr (DIM == 2) {
      int i = 0;
      for (k[0] = 0; k[0] < hooke.grid.N[0]; k[0]++) {
        for (k[1] = 0; k[1] < hooke.grid.N[1]; k[1]++) {
          hooke.modal_stiffness(k, K_k.data());
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
      int i = 0;
      for (k[0] = 0; k[0] < hooke.grid.N[0]; k[0]++) {
        for (k[1] = 0; k[1] < hooke.grid.N[1]; k[1]++) {
          for (k[2] = 0; k[2] < hooke.grid.N[2]; k[2]++) {
            hooke.modal_stiffness(k, K_k.data());
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
    double cell_volume = 1.0;
    for (int i = 0; i < DIM; i++) {
      fftw_execute(idft_Ku[i]);
      cell_volume *= hooke.grid.L[i] / hooke.grid.N[i];
    }
    double correction = cell_volume / hooke.grid.num_cells;
    // The following correction is due to (1) the fact that FFTW's backward
    // Fourier transform returns the inverse DFT, scaled by the number of cells
    // (see FFTW's FAQ, 3.10), and (2) the fact that the so-called “modal
    // stiffness matrix” returns computes the strain energy, scaled by `|h|`
    // (the volume of the cells, see “Theory” in the documentation.
    for (int i = 0; i < num_dofs; i++) {
      Ku.cpp_data[i] *= correction;
    }
  };

 public:
  StiffnessMatrixFactory(bri17::Hooke<T, DIM> hooke)
      : num_dofs{DIM * hooke.grid.num_cells},
        hooke{hooke},
        u{num_dofs},
        u_hat{num_dofs},
        Ku{num_dofs},
        Ku_hat{num_dofs} {
    int N_[DIM];
    for (int i = 0; i < DIM; i++) N_[i] = int(hooke.grid.N[i]);
    for (int i = 0; i < DIM; i++) {
      int offset = i * hooke.grid.num_cells;
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
    for (int i = 0; i < num_dofs; i++) {
      u.cpp_data[i] = 0;
    }
    for (int j = 0; j < num_dofs; j++) {
      u.cpp_data[j] = 1;
      compute_Ku();
      for (int i = 0; i < num_dofs; i++) {
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

template <int DIM>
Eigen::MatrixXd assemble_expected_stiffness_matrix(
    const bri17::CartesianGrid<double, DIM> &grid, const Eigen::MatrixXd &Ke) {
  // TODO Check dimensions of Ke and K.
  const int num_dofs_per_cell = grid.num_nodes_per_cell * DIM;
  const int num_dofs = grid.num_cells * DIM;
  Eigen::MatrixXd K{num_dofs, num_dofs};
  K.setZero();
  auto cell_nodes = new int[grid.num_nodes_per_cell];
  for (int cell = 0; cell < grid.num_cells; cell++) {
    grid.get_cell_nodes(cell, cell_nodes);
    for (int ie = 0; ie < num_dofs_per_cell; ie++) {
      int i = cell_nodes[ie % grid.num_nodes_per_cell] +
              grid.num_cells * (ie / grid.num_nodes_per_cell);
      for (int je = 0; je < num_dofs_per_cell; je++) {
        int j = cell_nodes[je % grid.num_nodes_per_cell] +
                grid.num_cells * (je / grid.num_nodes_per_cell);
        K(i, j) += Ke(ie, je);
      }
    }
  }
  delete[] cell_nodes;
  return K;
}

template <int DIM>
constexpr int num_strain_components() {
  return (DIM * (DIM + 1)) / 2;
}

template <typename T, int DIM>
class StrainDisplacementMatrixFactory {
 private:
  const int num_dofs;
  const bri17::Hooke<T, DIM> hooke;
  const FFTWComplexBuffer u;
  const FFTWComplexBuffer u_hat;
  const FFTWComplexBuffer Bu;
  const FFTWComplexBuffer Bu_hat;
  // TODO These members should be const
  fftw_plan dft_u[DIM];
  fftw_plan idft_Bu[num_strain_components<DIM>()];

  void compute_Bu() {
    for (int i = 0; i < DIM; i++) fftw_execute(dft_u[i]);
    int k[DIM] = {0};
    Eigen::Matrix<std::complex<double>, DIM, 1> B_k;
    Eigen::Matrix<std::complex<double>, DIM, 1> u_k;
    if constexpr (DIM == 2) {
      int i = 0;
      for (k[0] = 0; k[0] < hooke.grid.N[0]; k[0]++) {
        for (k[1] = 0; k[1] < hooke.grid.N[1]; k[1]++) {
          hooke.modal_strain_displacement(k, B_k.data());
          u_k(0) = u_hat.cpp_data[i];
          u_k(1) = u_hat.cpp_data[i + hooke.grid.num_cells];
          auto eps_k = 0.5 * (B_k * u_k.transpose() + u_k * B_k.transpose());
          Bu_hat.cpp_data[i] = eps_k(0, 0);
          Bu_hat.cpp_data[i + hooke.grid.num_cells] = eps_k(1, 1);
          Bu_hat.cpp_data[i + 2 * hooke.grid.num_cells] = sqrt(2) * eps_k(0, 1);
          i++;
        }
      }
    } else if constexpr (DIM == 3) {
      // TODO: the two cases should be merged
      int i = 0;
      for (k[0] = 0; k[0] < hooke.grid.N[0]; k[0]++) {
        for (k[1] = 0; k[1] < hooke.grid.N[1]; k[1]++) {
          for (k[2] = 0; k[2] < hooke.grid.N[2]; k[2]++) {
            hooke.modal_strain_displacement(k, B_k.data());
            u_k(0) = u_hat.cpp_data[i];
            u_k(1) = u_hat.cpp_data[i + hooke.grid.num_cells];
            u_k(2) = u_hat.cpp_data[i + 2 * hooke.grid.num_cells];
            auto eps_k = 0.5 * (B_k * u_k.transpose() + u_k * B_k.transpose());
            Bu_hat.cpp_data[i] = eps_k(0, 0);
            Bu_hat.cpp_data[i + hooke.grid.num_cells] = eps_k(1, 1);
            Bu_hat.cpp_data[i + 2 * hooke.grid.num_cells] = eps_k(2, 2);

            Bu_hat.cpp_data[i + 3 * hooke.grid.num_cells] =
                sqrt(2) * eps_k(1, 2);
            Bu_hat.cpp_data[i + 4 * hooke.grid.num_cells] =
                sqrt(2) * eps_k(2, 0);
            Bu_hat.cpp_data[i + 5 * hooke.grid.num_cells] =
                sqrt(2) * eps_k(0, 1);
            i++;
          }
        }
      }
    }
    for (int i = 0; i < num_strain_components<DIM>(); i++)
      fftw_execute(idft_Bu[i]);
    double correction = 1.0 / hooke.grid.num_cells;
    // The following correction is due to the fact that FFTW's backward Fourier
    // transform returns the inverse DFT, scaled by the number of cells (see
    // FFTW's FAQ, 3.10).
    for (int i = 0; i < num_strain_components<DIM>() * hooke.grid.num_cells;
         i++) {
      Bu.cpp_data[i] *= correction;
    }
  };

 public:
  StrainDisplacementMatrixFactory(bri17::Hooke<T, DIM> hooke)
      : num_dofs{DIM * hooke.grid.num_cells},
        hooke{hooke},
        u{num_dofs},
        u_hat{num_dofs},
        Bu{num_strain_components<DIM>() * hooke.grid.num_cells},
        Bu_hat{num_strain_components<DIM>() * hooke.grid.num_cells} {
    int N_[DIM];
    for (int i = 0; i < DIM; i++) N_[i] = int(hooke.grid.N[i]);
    for (int i = 0; i < DIM; i++) {
      int offset = i * hooke.grid.num_cells;
      dft_u[i] =
          fftw_plan_dft(DIM, N_, u.c_data + offset, u_hat.c_data + offset,
                        FFTW_FORWARD, FFTW_ESTIMATE);
    }
    for (int i = 0; i < num_strain_components<DIM>(); i++) {
      int offset = i * hooke.grid.num_cells;
      idft_Bu[i] =
          fftw_plan_dft(DIM, N_, Bu_hat.c_data + offset, Bu.c_data + offset,
                        FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }

  Eigen::MatrixXd run() {
    const int num_rows = num_strain_components<DIM>() * hooke.grid.num_cells;
    Eigen::MatrixXd B{num_rows, num_dofs};
    for (int i = 0; i < num_dofs; i++) {
      u.cpp_data[i] = 0;
    }
    for (int j = 0; j < num_dofs; j++) {
      u.cpp_data[j] = 1;
      compute_Bu();
      for (int i = 0; i < num_rows; i++) {
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

template <typename T, int DIM>
Eigen::MatrixXd assemble_expected_strain_displacement_matrix(
    const bri17::CartesianGrid<T, DIM> &grid, const Eigen::MatrixXd &Be) {
  const int num_strain_components = (DIM * (DIM + 1)) / 2;
  const int num_rows = grid.num_cells * num_strain_components;
  const int num_dofs_per_cell = grid.num_nodes_per_cell * DIM;
  const int num_dofs = grid.num_cells * DIM;
  Eigen::MatrixXd B{num_rows, num_dofs};
  B.setZero();
  auto cell_nodes = new int[grid.num_nodes_per_cell];
  for (int cell = 0; cell < grid.num_cells; cell++) {
    grid.get_cell_nodes(cell, cell_nodes);
    for (int i_local = 0; i_local < num_strain_components; i_local++) {
      int i = i_local * grid.num_cells + cell;
      for (int j_local = 0; j_local < num_dofs_per_cell; j_local++) {
        int j = cell_nodes[j_local % grid.num_nodes_per_cell] +
                grid.num_cells * (j_local / grid.num_nodes_per_cell);
        B(i, j) += Be(i_local, j_local);
      }
    }
  }
  delete[] cell_nodes;
  return B;
}

TEST_CASE("Global assembly tests") {
  constexpr int max_dim = 3;
  const double mu = 5.6;
  const double nu = 0.3;

  std::array<int, 2> shape2{3, 4};
  std::array<double, 2> spacing2{1.1, 1.2};
  std::array<double, 2> L2;
  std::transform(shape2.cbegin(), shape2.cend(), spacing2.cbegin(), L2.begin(),
                 std::multiplies());

  std::array<int, 3> shape3{3, 4, 5};
  std::array<double, 3> spacing3{1.1, 1.2, 1.3};
  std::array<double, 3> L3;
  std::transform(shape3.cbegin(), shape3.cend(), spacing3.cbegin(), L3.begin(),
                 std::multiplies());

  SECTION("2D stiffness matrix") {
    const int dim = 2;
    bri17::CartesianGrid<decltype(L2)::value_type, dim> grid{shape2, L2};
    bri17::Hooke hooke{mu, nu, grid};
    StiffnessMatrixFactory factory{hooke};

    const int num_dofs_per_cell = grid.num_nodes_per_cell * dim;
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
    const int dim = 3;
    bri17::CartesianGrid<decltype(L3)::value_type, dim> grid{shape3, L3};
    bri17::Hooke hooke{mu, nu, grid};
    StiffnessMatrixFactory factory{hooke};

    const int num_dofs_per_cell = grid.num_nodes_per_cell * dim;
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
    const int dim = 2;
    bri17::CartesianGrid<decltype(L2)::value_type, dim> grid{shape2, L2};
    bri17::Hooke hooke{mu, nu, grid};

    const int num_dofs_per_cell = grid.num_nodes_per_cell * dim;
    const int num_strain_components_per_cell = (dim * (dim + 1)) / 2;
    Eigen::MatrixXd Be{num_strain_components_per_cell, num_dofs_per_cell};
    // This is a copy-paste from Maxima
    Be << -0.6, -0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.55,
        0.55, -0.55, 0.55, -0.3889087296526009, 0.3889087296526009,
        -0.3889087296526009, 0.3889087296526009, -0.4242640687119284,
        -0.4242640687119284, 0.4242640687119284, 0.4242640687119284;
    auto B_exp = assemble_expected_strain_displacement_matrix(grid, Be);

    StrainDisplacementMatrixFactory factory{hooke};
    auto B_act = factory.run();

    assert_equal(B_exp, B_act, 1e-15, 1e-14);
  }

  SECTION("3D strain-displacement matrix") {
    const int dim = 3;
    bri17::CartesianGrid<decltype(L3)::value_type, dim> grid{shape3, L3};
    bri17::Hooke hooke{mu, nu, grid};

    const int num_dofs_per_cell = grid.num_nodes_per_cell * dim;
    const int num_strain_components_per_cell = (dim * (dim + 1)) / 2;
    Eigen::MatrixXd Be{num_strain_components_per_cell, num_dofs_per_cell};
    // This is a copy-paste from Maxima
    Be << -0.39, -0.39, -0.39, -0.39, 0.39, 0.39, 0.39, 0.39, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3575, -0.3575, 0.3575, 0.3575,
        -0.3575, -0.3575, 0.3575, 0.3575, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, -0.33, 0.33, -0.33, 0.33, -0.33, 0.33, -0.33, 0.33, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2333452377915605,
        0.2333452377915605, -0.2333452377915605, 0.2333452377915605,
        -0.2333452377915605, 0.2333452377915605, -0.2333452377915605,
        0.2333452377915605, -0.2527906742741904, -0.2527906742741904,
        0.2527906742741904, 0.2527906742741904, -0.2527906742741904,
        -0.2527906742741904, 0.2527906742741904, 0.2527906742741904,
        -0.2333452377915605, 0.2333452377915605, -0.2333452377915605,
        0.2333452377915605, -0.2333452377915605, 0.2333452377915605,
        -0.2333452377915605, 0.2333452377915605, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -0.2757716446627534, -0.2757716446627534, -0.2757716446627534,
        -0.2757716446627534, 0.2757716446627534, 0.2757716446627534,
        0.2757716446627534, 0.2757716446627534, -0.2527906742741904,
        -0.2527906742741904, 0.2527906742741904, 0.2527906742741904,
        -0.2527906742741904, -0.2527906742741904, 0.2527906742741904,
        0.2527906742741904, -0.2757716446627534, -0.2757716446627534,
        -0.2757716446627534, -0.2757716446627534, 0.2757716446627534,
        0.2757716446627534, 0.2757716446627534, 0.2757716446627534, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    auto B_exp = assemble_expected_strain_displacement_matrix(grid, Be);

    StrainDisplacementMatrixFactory factory{hooke};
    auto B_act = factory.run();

    assert_equal(B_exp, B_act, 1e-15, 1e-14);
  }
}
