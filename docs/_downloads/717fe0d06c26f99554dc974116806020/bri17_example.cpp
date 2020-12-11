#include <iostream>
#include "bri17/bri17.hpp"

int main() {
  size_t N[] = {3, 4, 5};
  double L[] = {1., 2., 3.};
  bri17::CartesianGrid<3> grid{N, L};
  std::cout << grid << std::endl;
  return 0;
}
