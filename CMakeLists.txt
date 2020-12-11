cmake_minimum_required(VERSION 3.13)
project(
  bri17
  VERSION 0.1
  DESCRIPTION "Implementation of the matrix-free finite element method introduced in <https://hal-enpc.archives-ouvertes.fr/hal-01304603>"
  LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(
  ${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

if (MATH_LIBRARY)
  target_link_libraries(${PROJECT_NAME} INTERFACE ${MATH_LIBRARY})
endif()
add_compile_definitions(_USE_MATH_DEFINES)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}_Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

write_basic_package_version_file(
  "${PROJECT_NAME}-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME})

install(
  EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME})

install(
  FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/bri17 TYPE INCLUDE)

add_subdirectory(test)
