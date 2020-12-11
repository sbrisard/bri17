add_executable(test_bri17 test_bri17_main.cpp test_bri17.cpp ../include/bri17/bri17.hpp)

target_include_directories(test_bri17 PUBLIC ../include)

if (MATH_LIBRARY)
  target_link_libraries(test_bri17 ${MATH_LIBRARY})
endif()

find_package(Catch2 REQUIRED)
target_link_libraries(test_bri17 Catch2::Catch2)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(test_bri17 Eigen3::Eigen)

find_path(FFTW3_INCLUDE_DIR fftw3.h)
find_library(FFTW3_LIBRARY NAMES fftw3 libfftw3 libfftw3-3)
target_link_libraries(test_bri17 ${FFTW3_LIBRARY})

add_test(NAME test_bri17 COMMAND test_bri17)
