cmake_minimum_required(VERSION 3.13)

file(READ ./metadata/version.txt METADATA_VERSION)
string(STRIP ${METADATA_VERSION} METADATA_VERSION)
file(READ ./metadata/description.txt METADATA_DESCRIPTION)
string(STRIP ${METADATA_DESCRIPTION} METADATA_DESCRIPTION)
file(READ ./metadata/author.txt METADATA_AUTHOR)
string(STRIP ${METADATA_AUTHOR} METADATA_AUTHOR)

project(
  bri17
  VERSION ${METADATA_VERSION}
  DESCRIPTION ${METADATA_AUTHOR}
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(
  ${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

#target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

if (MATH_LIBRARY)
  target_link_libraries(${PROJECT_NAME} INTERFACE ${MATH_LIBRARY})
endif()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}_Targets)

install(
  EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

write_basic_package_version_file(
  "${PROJECT_NAME}-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})


install(
  FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/bri17 TYPE INCLUDE)

add_subdirectory(tests)
add_subdirectory(python)