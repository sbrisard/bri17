cmake_minimum_required(VERSION 3.13)
project(bri17 CXX)

set(CMAKE_CXX_STANDARD 20)
include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules/morse_cmake/modules" )
include(MorseInit)

add_subdirectory(include)
add_subdirectory(test)

# TODO: this should be defined as a variable
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs/")

set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_HTML_OUTPUT "${CMAKE_SOURCE_DIR}/docs")
set(DOXYGEN_GENERATE_LATEX YES)
set(DOXYGEN_LATEX_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/docs/latex")

find_package(Doxygen)
doxygen_add_docs(
  doxygen
  "${CMAKE_SOURCE_DIR}/include/bri17"
  ALL
  COMMENT "Generate docs")
