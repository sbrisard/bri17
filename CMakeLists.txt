cmake_minimum_required(VERSION 3.13)
project(bri17 CXX)

set(CMAKE_CXX_STANDARD 20)
include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules/morse_cmake/modules" )
include(MorseInit)

add_subdirectory(src)
# add_subdirectory(test)
