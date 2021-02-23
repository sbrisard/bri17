#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bri17/bri17.hpp"

namespace py = pybind11;

template <typename T>
void check_contiguous(py::array_t<T> array) {
  auto info = array.request();
  if ((info.ndim != 1) || (info.strides[0] != sizeof(T))) {
    throw std::invalid_argument("expected one-dimensional, contiguous array");
  }
}

template <typename T, int SIZE>
requires((SIZE >= 0) &&
         (SIZE < 5)) auto make_tuple_from_std_array(std::array<T, SIZE> a) {
  if constexpr (SIZE == 0)
    return py::make_tuple();
  else if constexpr (SIZE == 1)
    return py::make_tuple(a[0]);
  else if constexpr (SIZE == 2)
    return py::make_tuple(a[0], a[1]);
  else if constexpr (SIZE == 3)
    return py::make_tuple(a[0], a[1], a[2]);
  else if constexpr (SIZE == 4)
    return py::make_tuple(a[0], a[1], a[2], a[3]);
}

template <typename T, int DIM>
void create_cartesian_grid_binding(py::module m, const char* name) {
  using CartesianGrid = bri17::CartesianGrid<T, DIM>;

  auto py_class_ = py::class_<CartesianGrid>(m, name);
  py_class_.def(py::init<std::array<int, DIM>, std::array<T, DIM>>())
      .def_property_readonly_static(
          "dtype", [](py::object) { return py::dtype::of<T>(); })
      .def_property_readonly_static("dim", [](py::object) { return DIM; })
      .def_property_readonly("shape",
                             [](CartesianGrid& self) {
                               return make_tuple_from_std_array(self.shape);
                             })
      .def_readonly("size", &CartesianGrid::size)
      .def_property_readonly(
          "L",
          [](CartesianGrid& self) { return make_tuple_from_std_array(self.L); })
      .def("__repr__", &CartesianGrid::repr)
      .def("get_cell_nodes", &CartesianGrid::get_cell_nodes);

  if constexpr (DIM == 2) {
    py_class_.def("get_node_at", [](CartesianGrid& self, int i, int j) {
      return self.get_node_at(i, j);
    });
  } else if constexpr (DIM == 3) {
    py_class_.def("get_node_at", [](CartesianGrid& self, int i, int j, int k) {
      return self.get_node_at(i, j, k);
    });
  }
}

template <typename T, int DIM>
void create_hooke_binding(py::module m, const char* name) {
  using CartesianGrid = bri17::CartesianGrid<T, DIM>;
  using Hooke = bri17::Hooke<T, DIM>;
  using int_array_t = py::array_t<int>;
  using complex_array_t = py::array_t<std::complex<T>>;

  auto py_class_ = py::class_<Hooke>(m, name);
  py_class_.def(py::init<T, T, CartesianGrid>())
      .def_readonly("mu", &Hooke::mu)
      .def_readonly("nu", &Hooke::nu)
      .def_readonly("grid", &Hooke::grid)
      .def("__repr__", &Hooke::repr)
      .def("modal_strain_displacement",
           [](Hooke& self, int_array_t k, complex_array_t B) {
             check_contiguous(k);
             check_contiguous(B);
             return self.modal_strain_displacement(k.data(), B.mutable_data());
           })
      .def("modal_stiffness_matrix",
           [](Hooke& self, int_array_t k, complex_array_t K) {
             check_contiguous(k);
             check_contiguous(K);
             return self.modal_stiffness(k.data(), K.mutable_data());
           })
      .def("modal_eigenstress_to_opposite_strain",
           [](Hooke& self, int_array_t k, complex_array_t tau,
              complex_array_t eta) {
             check_contiguous(tau);
             check_contiguous(eta);
             return self.modal_eigenstress_to_opposite_strain(
                 k.data(), tau.data(), eta.mutable_data());
           });
}

PYBIND11_MODULE(pybri17, m) {
  // m.doc() = py::cast(__DOC__);
  m.attr("__author__") = py::cast(__AUTHOR__);
  m.attr("__version__") = py::cast(__VERSION__);

  create_cartesian_grid_binding<double, 2>(m, "CartesianGrid2f64");
  create_cartesian_grid_binding<double, 3>(m, "CartesianGrid3f64");
  create_hooke_binding<double, 2>(m, "Hooke2f64");
  create_hooke_binding<double, 3>(m, "Hooke3f64");
}
