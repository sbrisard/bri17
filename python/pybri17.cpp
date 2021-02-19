#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bri17/bri17.hpp"

namespace py = pybind11;

// template <typename T, int DIM>

template <typename T, int DIM>
auto create_binding(py::module m, const char* name) {
  using CartesianGrid = bri17::CartesianGrid<T, DIM>;
  return py::class_<Hooke>(m, name).def(py::init<>());
}

PYBIND11_MODULE(pybri17, m) {
  // m.doc() = py::cast(__DOC__);
  m.attr("__author__") = py::cast(__AUTHOR__);
  m.attr("__version__") = py::cast(__VERSION__);
}
