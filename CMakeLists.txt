cmake_minimum_required(VERSION 3.13)
project(bri17 CXX)

set(CMAKE_CXX_STANDARD 20)
include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules/morse_cmake/modules" )
include(MorseInit)

add_subdirectory(include)
add_subdirectory(test)

set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_XML_OUTPUT "${CMAKE_SOURCE_DIR}/sphinx/doxyxml")

find_package(Doxygen)
doxygen_add_docs(
  doxygen
  "${CMAKE_SOURCE_DIR}/include/bri17"
  ALL
  COMMENT "Generate docs")
