add_library(bri17 INTERFACE)

target_include_directories(bri17 INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/bri17 TYPE INCLUDE)
