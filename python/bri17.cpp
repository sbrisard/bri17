#include <pybind11/pybind11.h>

#include "bri17/bri17.hpp"

PYBIND11_MODULE(bri17, m) {
  m.doc() = __DOC__;
  m.attr("__author__") = __AUTHOR__;
  m.attr("__version__") = __VERSION__;
}
